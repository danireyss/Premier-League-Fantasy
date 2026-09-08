"""Ingest daemon. Run this as a separate long-lived process from Streamlit.

    uv run flive-ingest

Streamlit reruns its whole script on every widget interaction and every
autorefresh, so a poller living inside the app would be restarted constantly,
duplicate itself across sessions, and hammer an API that has no SLA. Keep them
separate: this process writes, Streamlit only ever reads.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from . import parse, store
from .clients import FPL
from .config import FLUSH_EVERY, IDLE_INTERVAL, LIVE_INTERVAL, SLOW_INTERVAL

log = logging.getLogger("flive.ingest")
_stop = asyncio.Event()

# Written by slow_loop, read by live_loop. bootstrap-static is the only place
# that knows a player's club and position, and live_loop needs both to attribute
# a stat line to a side. Refreshed every SLOW_INTERVAL; players do not move club
# mid-match.
_meta: dict[int, dict] = {}
_team_names: dict[int, str] = {}


async def live_loop(fpl: FPL) -> None:
    """Gameweek points, xG and match state — every 60s while matches are on."""
    fixtures_buf = store.Buffer("fixture_snapshots", FLUSH_EVERY)
    players_buf = store.Buffer("player_snapshots", FLUSH_EVERY)

    while not _stop.is_set():
        try:
            bootstrap_gw = await _current_gw(fpl)
            if bootstrap_gw is None:
                log.info("no current gameweek, idling %ss", IDLE_INTERVAL)
                await _sleep_or_stop(IDLE_INTERVAL)
                continue

            fixtures = await fpl.fixtures(bootstrap_gw)
            in_play = [f for f in fixtures if f.get("started") and not f.get("finished")]

            if not in_play:
                # Nothing is on. Polling hard against an unsupported API that
                # will not change for hours is how you get your IP throttled.
                for buf in (fixtures_buf, players_buf):
                    buf.maybe_flush(force=True)
                log.info("gw%d: nothing in play, idling %ss", bootstrap_gw, IDLE_INTERVAL)
                await _sleep_or_stop(IDLE_INTERVAL)
                continue

            status = await fpl.event_status()
            final = parse.bonus_is_final(status)
            live = await fpl.live(bootstrap_gw)

            captured_at = store.now()
            team_fixture = parse.team_fixture_map(fixtures)
            team_xg = parse.team_xg_from_players(live, _meta, team_fixture)

            players_buf.add(
                parse.player_rows(
                    live, bootstrap_gw, final, _meta, team_fixture, captured_at
                )
            )
            fixtures_buf.add(
                parse.fixture_rows(fixtures, team_xg, _team_names, captured_at)
            )

            for buf in (fixtures_buf, players_buf):
                buf.maybe_flush()

            log.info(
                "gw%d tick: %d in play, bonus_final=%s", bootstrap_gw, len(in_play), final
            )
        except Exception:
            log.exception("live poll failed")

        await _sleep_or_stop(LIVE_INTERVAL)

    for buf in (fixtures_buf, players_buf):
        buf.maybe_flush(force=True)


async def slow_loop(fpl: FPL) -> None:
    """Prices, ownership, injuries and the player->club map, every 6h.

    bootstrap-static is a multi-megabyte payload containing every player in the
    game. Prices only change once a day, so polling it fast buys nothing and is
    the fastest way to get your IP throttled.
    """
    while not _stop.is_set():
        try:
            bootstrap = await fpl.bootstrap()
            _refresh_meta(bootstrap)
            captured_at = store.now()

            rows = parse.fpl_player_rows(bootstrap, captured_at)
            store.write("fpl_players", store.conform("fpl_players", rows))
            teams = parse.fpl_team_rows(bootstrap, captured_at)
            store.write("fpl_teams", store.conform("fpl_teams", teams))

            # The whole season's fixture list, not just this gameweek's. The
            # projection needs the unplayed ones to know who a club faces next
            # and the played ones to know how many matches its rates rest on.
            schedule = parse.schedule_rows(await fpl.schedule(), _team_names, captured_at)
            store.write("fixtures", store.conform("fixtures", schedule))

            log.info(
                "bootstrap: %d players, %d teams, %d fixtures",
                len(rows), len(teams), len(schedule),
            )
        except Exception:
            log.exception("bootstrap failed")
        await _sleep_or_stop(SLOW_INTERVAL)


def _refresh_meta(bootstrap: dict) -> None:
    _meta.clear()
    _meta.update(parse.player_meta(bootstrap))
    _team_names.clear()
    _team_names.update(parse.team_names(bootstrap))


async def _current_gw(fpl: FPL) -> int | None:
    bootstrap = await fpl.bootstrap()
    if not _meta:
        _refresh_meta(bootstrap)
    return parse.current_gameweek(bootstrap)


async def _sleep_or_stop(seconds: float) -> None:
    try:
        await asyncio.wait_for(_stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    fpl = FPL()

    # Prime the club/position map before the live loop needs it, so the first
    # tick can attribute stat lines instead of writing a batch of nulls.
    try:
        bootstrap = await fpl.bootstrap()
        _refresh_meta(bootstrap)
        log.info("primed %d players across %d teams", len(_meta), len(_team_names))
    except Exception:
        log.exception("could not prime player metadata; first ticks may be sparse")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop.set)

    try:
        await asyncio.gather(live_loop(fpl), slow_loop(fpl))
    finally:
        await fpl.aclose()
        log.info("stopped cleanly")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
