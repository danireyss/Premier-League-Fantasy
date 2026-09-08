"""Read side. Everything is a LazyFrame until the last moment.

The point of storing snapshots rather than current state is that these queries
become possible. FPL tells you a player has 0.61 xG. It will not tell you that
0.4 of it arrived in the last twelve minutes — that is yours to derive, and it
is the difference between a scoreboard and an analytical app.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from . import project, store

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


# Metrics the compare view ranks: label -> (column, per-90-able). Rates like
# points-per-game and ownership are already normalised, so dividing them by
# minutes again would be meaningless — hence the flag.
COMPARE_STATS: dict[str, tuple[str, bool]] = {
    "Points": ("total_points", True),
    "Goals": ("goals_scored", True),
    "Assists": ("assists", True),
    "xG": ("xg", True),
    "xA": ("xa", True),
    "xGI": ("xgi", True),
    "Threat": ("threat", True),
    "Creativity": ("creativity", True),
    "Influence": ("influence", True),
    "ICT": ("ict_index", True),
    "BPS": ("bps", True),
    "Defensive contribution": ("defensive_contribution", True),
    "Points per game": ("points_per_game", False),
    "Owned %": ("selected_by_pct", False),
}

# The subset drawn as percentile bars — the shape of a player rather than his
# volume. ICT is left out on purpose: it is the sum of the three beside it, so
# plotting it as a fourth axis would count the same performance twice.
PROFILE_STATS = [
    "xG",
    "xA",
    "Threat",
    "Creativity",
    "Influence",
    "BPS",
    "Defensive contribution",
]


def compare_board(min_minutes: int = 90, per_90: bool = False) -> pl.DataFrame:
    """Season board with each metric's value and its percentile among peers.

    Percentiles are what makes a comparison readable: 0.42 xA means nothing on
    its own, but 83rd among midfielders places it. The pool is players at the
    same position clearing `min_minutes` — ranking a defender's threat against a
    striker's would flatter every striker, and leaving cameos in the pool would
    let one bright ten minutes outrank a season.

    Every player is returned, including those short of the threshold; they
    simply carry a null percentile, so you can still put a fringe player's raw
    numbers next to a regular's without silently ranking him against nobody.
    """
    board = player_board()
    if board.is_empty():
        return board

    per_90_expr = pl.col("minutes") / 90
    values = []
    for column, scalable in COMPARE_STATS.values():
        if not (per_90 and scalable):
            values.append(pl.col(column).cast(pl.Float64).alias(f"v_{column}"))
        elif f"{column}_per_90" in board.columns:
            values.append(pl.col(f"{column}_per_90").alias(f"v_{column}"))
        else:
            values.append(
                pl.when(pl.col("minutes") >= 90)
                .then(pl.col(column) / per_90_expr)
                .otherwise(None)
                .alias(f"v_{column}")
            )
    board = board.with_columns(values)

    pool = board.filter(pl.col("minutes") >= max(min_minutes, 1))
    if pool.is_empty():
        return board.with_columns(
            [pl.lit(None, pl.Float64).alias(f"p_{c}") for c, _ in COMPARE_STATS.values()]
        )

    # Rank counts non-nulls only, so the divisor has to as well or a metric with
    # gaps would top out below 100.
    ranked = pool.select(
        "fpl_id",
        *[
            (
                pl.col(f"v_{column}").rank("average").over("position")
                / pl.col(f"v_{column}").count().over("position")
                * 100
            ).alias(f"p_{column}")
            for column, _ in COMPARE_STATS.values()
        ],
    )
    return board.join(ranked, on="fpl_id", how="left")


def _played_matches() -> pl.DataFrame:
    """team_id -> matches completed, from the fixture list.

    Has to come from fixtures, not from minutes: a club that has had someone
    sent off has played the same number of matches with fewer player-minutes,
    and every rate in the projection divides by this.
    """
    sched = latest_schedule()
    if sched.is_empty():
        return pl.DataFrame(schema={"team_id": pl.Int32, "matches": pl.UInt32})
    done = sched.filter(pl.col("finished"))
    if done.is_empty():
        return pl.DataFrame(schema={"team_id": pl.Int32, "matches": pl.UInt32})
    return (
        pl.concat(
            [
                done.select(pl.col("home_team_id").alias("team_id")),
                done.select(pl.col("away_team_id").alias("team_id")),
            ]
        )
        .group_by("team_id")
        .agg(pl.len().alias("matches"))
    )


def latest_schedule() -> pl.DataFrame:
    """The fixture list as of the most recent slow-loop write.

    The schedule is rewritten whole on every tick rather than appended to, so
    the newest capture is the truth — an earlier one may still show a fixture
    at a date it has since been moved from.
    """
    lf = store.scan("fixtures")
    latest = lf.select(pl.col("captured_at").max()).collect().item()
    if latest is None:
        return pl.DataFrame()
    return lf.filter(pl.col("captured_at") == latest).collect()


def upcoming_gameweeks(limit: int = 8) -> list[int]:
    """Gameweeks with at least one fixture still to be played."""
    sched = latest_schedule()
    if sched.is_empty():
        return []
    return (
        sched.filter(~pl.col("finished"))
        .select("gw")
        .unique()
        .sort("gw")
        .head(limit)["gw"]
        .to_list()
    )


def projections(gws: list[int] | None = None) -> pl.DataFrame:
    """Expected points per player per upcoming fixture.

    One row per fixture, so a double gameweek shows as two rows for the same
    player and a blank as none. The caller decides whether to sum them.
    """
    sched = latest_schedule()
    if sched.is_empty():
        return pl.DataFrame()

    upcoming = sched.filter(~pl.col("finished"))
    if gws:
        upcoming = upcoming.filter(pl.col("gw").is_in(gws))
    if upcoming.is_empty():
        return pl.DataFrame()

    players = player_board()
    if players.is_empty():
        return pl.DataFrame()

    return project.expected_points(players, upcoming, _played_matches())


def team_strength() -> pl.DataFrame:
    """The club ratings the projection is built on, for inspection.

    Worth surfacing rather than hiding: a projection that looks wrong is
    usually a club rating that looks wrong, and this is where you see it.
    """
    players = player_board()
    if players.is_empty():
        return pl.DataFrame()
    ratings = project.team_ratings(players, _played_matches())
    if ratings.is_empty():
        return ratings
    names = (
        players.select("team_id", "team_name").unique(subset=["team_id"])
    )
    return ratings.join(names, on="team_id", how="left").sort("attack", descending=True)


def next_kickoff() -> datetime | None:
    """When the next unplayed fixture starts, or None if the season is over."""
    sched = latest_schedule()
    if sched.is_empty():
        return None
    ahead = sched.filter(~pl.col("started") & pl.col("kickoff_time").is_not_null())
    if ahead.is_empty():
        return None
    return ahead["kickoff_time"].min()


def recent_results(limit: int = 10) -> pl.DataFrame:
    """Finished fixtures, most recent first.

    Read from the schedule rather than fixture_snapshots: the live loop only
    writes while a match is in progress, so between gameweeks the snapshot
    tables are empty and this is the only record of what has been played.
    """
    sched = latest_schedule()
    if sched.is_empty():
        return sched
    return (
        sched.filter(pl.col("finished"))
        .sort("kickoff_time", descending=True, nulls_last=True)
        .head(limit)
    )


def next_fixtures(limit: int = 10) -> pl.DataFrame:
    """Fixtures not yet kicked off, soonest first."""
    sched = latest_schedule()
    if sched.is_empty():
        return sched
    return (
        sched.filter(~pl.col("started"))
        .sort("kickoff_time", nulls_last=True)
        .head(limit)
    )
