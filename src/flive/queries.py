"""Read side. Everything is a LazyFrame until the last moment.

The point of storing snapshots rather than current state is that these queries
become possible. FPL tells you a player has 0.61 xG. It will not tell you that
0.4 of it arrived in the last twelve minutes — that is yours to derive, and it
is the difference between a scoreboard and an analytical app.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from . import store

# Stats that only ever increase within a gameweek, so last minus first over a
# window is the amount added during it.
CUMULATIVE = [
    "xg",
    "xa",
    "xgi",
    "bps",
    "total_points",
    "minutes",
    "threat",
    "creativity",
    "influence",
    "tackles",
    "recoveries",
    "cbi",
    "goals_scored",
    "assists",
]

LIVE_STATES = ["LIVE", "PROV"]


def _latest_per(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Most recent snapshot for each key."""
    return lf.sort("captured_at").group_by(keys, maintain_order=True).agg(pl.all().last())


def live_fixtures() -> pl.DataFrame:
    """Current state of every fixture we have seen, most recent first."""
    return (
        _latest_per(store.scan("fixture_snapshots"), ["fixture_id"])
        .with_columns(
            (pl.col("home_xg") - pl.col("away_xg")).alias("xg_diff"),
            (pl.col("home_score") - pl.col("away_score")).alias("goal_diff"),
        )
        .sort("captured_at", descending=True)
        .collect()
    )


def xg_timeline(fixture_id: int) -> pl.DataFrame:
    """Team xG as a stepped series.

    Plot with step interpolation, not linear — xG only moves when a shot is
    registered and processed, so a smooth line implies chances that did not
    happen. These totals are summed from the live per-player xG of each side.
    """
    return (
        store.scan("fixture_snapshots")
        .filter(pl.col("fixture_id") == fixture_id)
        .select("captured_at", "minute", "home_xg", "away_xg", "home_score", "away_score")
        .sort("captured_at")
        .unique(subset=["minute", "home_xg", "away_xg"], keep="first", maintain_order=True)
        .collect()
    )


def player_momentum(window_minutes: int = 15, min_snapshots: int = 2) -> pl.DataFrame:
    """How much each player has accumulated in the trailing window.

    Grouped by gameweek as well as player: totals reset to zero at a gameweek
    boundary, and a window straddling one would read that reset as a large
    negative move. Players who came off the pitch flatline and fall to the
    bottom on their own.
    """
    lf = store.scan("player_snapshots")
    cutoff = lf.select(pl.col("captured_at").max()).collect().item()
    if cutoff is None:
        return pl.DataFrame()
    start = cutoff - timedelta(minutes=window_minutes)

    windowed = lf.filter(pl.col("captured_at") >= start).sort("captured_at")

    return (
        windowed.group_by(["gw", "fpl_id"], maintain_order=True)
        .agg(
            pl.col("web_name").last(),
            pl.col("team_id").last(),
            pl.col("fixture_id").last(),
            pl.col("position").last(),
            pl.len().alias("snapshots"),
            *[
                (pl.col(c).drop_nulls().last() - pl.col(c).drop_nulls().first()).alias(f"d_{c}")
                for c in CUMULATIVE
            ],
            *[pl.col(c).drop_nulls().last().alias(c) for c in CUMULATIVE],
        )
        .filter(pl.col("snapshots") >= min_snapshots)
        .sort("d_xgi", descending=True, nulls_last=True)
        .collect()
    )


def fantasy_board(window_minutes: int = 15) -> pl.DataFrame:
    """Live FPL points beside the underlying performance driving them.

    One provider means one keyspace: the live stat line and the price/ownership
    metadata are both keyed by fpl_id, so this is a plain join. There is no
    cross-provider id mapping to maintain and nothing to re-run after a transfer
    window.
    """
    live = _latest_per(store.scan("player_snapshots"), ["fpl_id"]).collect()
    meta = _latest_per(store.scan("fpl_players"), ["fpl_id"]).collect()
    if live.is_empty() or meta.is_empty():
        return pl.DataFrame()

    momentum = player_momentum(window_minutes)

    board = live.select(
        "fpl_id", "gw", "web_name", "position", "team_id", "minutes", "total_points",
        "bonus", "bps", "goals_scored", "assists", "xg", "xa", "xgi", "threat",
        "bonus_final",
    ).join(
        meta.select("fpl_id", "team_name", "now_cost", "selected_by_pct", "form", "status"),
        on="fpl_id",
        how="left",
    )

    if not momentum.is_empty():
        board = board.join(
            momentum.select(["fpl_id", "d_xgi", "d_xg", "d_bps", "d_total_points"]),
            on="fpl_id",
            how="left",
        )

    return board.with_columns(
        (pl.col("now_cost") / 10).alias("price"),
        (pl.col("total_points") / (pl.col("now_cost") / 10)).alias("points_per_million"),
    ).sort("total_points", descending=True, nulls_last=True)


def player_board() -> pl.DataFrame:
    """Season-to-date totals for every player, one row each.

    Reads the slow loop's bootstrap snapshots rather than the live table: these
    are accumulated season figures, and only bootstrap-static carries them.

    A note on key passes: FPL publishes no raw key-pass count — no pass, shot or
    chance-creation counter exists anywhere in the API. `creativity` is Opta's
    chance-creation index and `xa` the expected value of chances created; those
    two are the honest stand-ins, and xa is arguably the better metric anyway
    since it weights the quality of a chance rather than counting it.
    """
    board = _latest_per(store.scan("fpl_players"), ["fpl_id"]).collect()
    if board.is_empty():
        return board

    per_90 = pl.col("minutes") / 90
    return board.with_columns(
        (pl.col("now_cost") / 10).alias("price"),
        # FPL ships per-90s for xG/xA/xGI only; derive the rest ourselves, and
        # only where a player has actually played enough for it to mean anything.
        *[
            pl.when(pl.col("minutes") >= 90)
            .then(pl.col(c) / per_90)
            .otherwise(None)
            .alias(f"{c}_per_90")
            for c in ("ict_index", "creativity", "threat", "influence")
        ],
    ).sort("total_points", descending=True, nulls_last=True)


def price_watch(hours: int = 24) -> pl.DataFrame:
    """Players whose price or ownership has moved since the oldest snapshot in
    the window. Only the slow loop feeds this, so it updates a few times a day."""
    lf = store.scan("fpl_players")
    latest_ts = lf.select(pl.col("captured_at").max()).collect().item()
    if latest_ts is None:
        return pl.DataFrame()
    start = latest_ts - timedelta(hours=hours)

    return (
        lf.filter(pl.col("captured_at") >= start)
        .sort("captured_at")
        .group_by("fpl_id", maintain_order=True)
        .agg(
            pl.col("web_name").last(),
            pl.col("position").last(),
            pl.col("team_name").last(),
            (pl.col("now_cost").last() - pl.col("now_cost").first()).alias("cost_delta"),
            (pl.col("selected_by_pct").last() - pl.col("selected_by_pct").first()).alias(
                "ownership_delta"
            ),
            (pl.col("now_cost").last() / 10).alias("price"),
            pl.col("selected_by_pct").last().alias("ownership"),
        )
        .filter((pl.col("cost_delta") != 0) | (pl.col("ownership_delta").abs() > 0.5))
        .sort("ownership_delta", descending=True, nulls_last=True)
        .collect()
    )


def coverage() -> dict[str, int]:
    """Row counts per table, for the status strip. Cheap: parquet metadata only."""
    out = {}
    for table in store.SCHEMAS:
        try:
            out[table] = store.scan(table).select(pl.len()).collect().item()
        except Exception:
            out[table] = 0
    return out
