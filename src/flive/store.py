"""Append-only parquet store, hive-partitioned by date.

Why parquet and not DuckDB: DuckDB takes an exclusive lock on its file for a
writing process. The ingest daemon and the Streamlit app are separate processes,
so a shared .duckdb file means Streamlit either fails to open it or blocks the
writer. Hive-partitioned parquet gives you one writer and unlimited concurrent
readers for free, and pl.scan_parquet pushes filters and projections down into
the files, so it stays fast well past the volume this app will ever produce.

Nothing here ever updates a row. Every tick is a new snapshot. The time series
is the whole point: FPL tells you a player's current total and keeps no history
of its own live state, so you build it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .config import DATA_DIR

TS = pl.Datetime("us", "UTC")

# Explicit schemas matter more than usual here. If a tick arrives with all-null
# xg, Polars would infer Null dtype, write that to parquet, and every later scan
# across the glob would fail on schema mismatch. Pinning the schema avoids it.
SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "fixture_snapshots": {
        "captured_at": TS,
        "gw": pl.Int32,
        "fixture_id": pl.Int32,
        "kickoff_time": TS,
        "state": pl.Utf8,
        "minute": pl.Int32,
        "home_team_id": pl.Int32,
        "away_team_id": pl.Int32,
        "home_name": pl.Utf8,
        "away_name": pl.Utf8,
        "home_score": pl.Int32,
        "away_score": pl.Int32,
        # Summed from the live per-player xG of each side — FPL publishes no
        # team-level figure of its own.
        "home_xg": pl.Float64,
        "away_xg": pl.Float64,
    },
    "player_snapshots": {
        "captured_at": TS,
        "gw": pl.Int32,
        "fpl_id": pl.Int32,
        "fixture_id": pl.Int32,
        "team_id": pl.Int32,
        "web_name": pl.Utf8,
        "position": pl.Utf8,
        "minutes": pl.Int32,
        "total_points": pl.Int32,
        "bonus": pl.Int32,
        "bps": pl.Int32,
        "goals_scored": pl.Int32,
        "assists": pl.Int32,
        "clean_sheets": pl.Int32,
        "goals_conceded": pl.Int32,
        "own_goals": pl.Int32,
        "saves": pl.Int32,
        "yellow_cards": pl.Int32,
        "red_cards": pl.Int32,
        "tackles": pl.Int32,
        "recoveries": pl.Int32,
        "cbi": pl.Int32,
        "defensive_contribution": pl.Int32,
        "starts": pl.Int32,
        "xg": pl.Float64,
        "xa": pl.Float64,
        "xgi": pl.Float64,
        "xgc": pl.Float64,
        "influence": pl.Float64,
        "creativity": pl.Float64,
        "threat": pl.Float64,
        "ict_index": pl.Float64,
        "bonus_final": pl.Boolean,
    },
    # Season-to-date totals, not a live line. bootstrap-static is the only
    # endpoint carrying a player's accumulated xG/ICT, and it is what the
    # Players tab reads.
    "fpl_players": {
        "captured_at": TS,
        "fpl_id": pl.Int32,
        "web_name": pl.Utf8,
        "full_name": pl.Utf8,
        "team_id": pl.Int32,
        "team_name": pl.Utf8,
        "position": pl.Utf8,
        "now_cost": pl.Int32,
        "selected_by_pct": pl.Float64,
        "form": pl.Float64,
        "total_points": pl.Int32,
        "status": pl.Utf8,
        "chance_next_round": pl.Int32,
        # --- season totals ---
        "minutes": pl.Int32,
        "starts": pl.Int32,
        "goals_scored": pl.Int32,
        "assists": pl.Int32,
        "bonus": pl.Int32,
        "bps": pl.Int32,
        "xg": pl.Float64,
        "xa": pl.Float64,
        "xgi": pl.Float64,
        "xgc": pl.Float64,
        # ICT and its three components. creativity is the closest thing FPL
        # publishes to key passes — it is Opta's chance-creation index, and no
        # raw key-pass count exists anywhere in this API.
        "influence": pl.Float64,
        "creativity": pl.Float64,
        "threat": pl.Float64,
        "ict_index": pl.Float64,
        # --- per 90, as FPL computes them ---
        "xg_per_90": pl.Float64,
        "xa_per_90": pl.Float64,
        "xgi_per_90": pl.Float64,
        # --- defensive ---
        "tackles": pl.Int32,
        "recoveries": pl.Int32,
        "cbi": pl.Int32,
        "defensive_contribution": pl.Int32,
        # --- value ---
        "points_per_game": pl.Float64,
        "value_season": pl.Float64,
        "ep_next": pl.Float64,
        # --- the projection's inputs -------------------------------------
        # Everything below feeds project.py. Season counting stats first:
        # every scoring event FPL pays or docks points for, so a projection
        # can be built from rates rather than from points-per-game, which
        # tells you what happened but not what drove it.
        "clean_sheets": pl.Int32,
        "goals_conceded": pl.Int32,
        "own_goals": pl.Int32,
        "saves": pl.Int32,
        "yellow_cards": pl.Int32,
        "red_cards": pl.Int32,
        "penalties_saved": pl.Int32,
        "penalties_missed": pl.Int32,
        # Rates FPL publishes directly. xgc_per_90 is the team's expected
        # goals conceded while this player was on the pitch, which is the
        # cleanest defensive read the API offers.
        "xgc_per_90": pl.Float64,
        "saves_per_90": pl.Float64,
        "goals_conceded_per_90": pl.Float64,
        "starts_per_90": pl.Float64,
        "clean_sheets_per_90": pl.Float64,
        "defensive_contribution_per_90": pl.Float64,
        # Set-piece and availability context. penalties_order is 1 for a
        # club's first-choice taker and null for everyone else.
        "penalties_order": pl.Int32,
        "corners_order": pl.Int32,
        "freekicks_order": pl.Int32,
        "news": pl.Utf8,
        "ep_this": pl.Float64,
    },
    # The schedule, not the live state — every fixture of the season including
    # ones not yet played. fixture_snapshots is a time series of matches in
    # progress; this is the fixture list, rewritten by the slow loop, and it is
    # what a projection needs: who a team plays next, where, and how hard FPL
    # rates it.
    "fixtures": {
        "captured_at": TS,
        "fixture_id": pl.Int32,
        "gw": pl.Int32,
        "kickoff_time": TS,
        "home_team_id": pl.Int32,
        "away_team_id": pl.Int32,
        "home_name": pl.Utf8,
        "away_name": pl.Utf8,
        # FPL's own fixture difficulty rating, 1 (easiest) to 5, set per side
        # and already venue-aware — the home and away figures differ.
        "home_difficulty": pl.Int32,
        "away_difficulty": pl.Int32,
        # Null until the match is played. Carried here so the app has something
        # to show between gameweeks: the live tables are written only while a
        # match is in progress, so for the four days between one gameweek and
        # the next they are empty and the schedule is all there is.
        "home_score": pl.Int32,
        "away_score": pl.Int32,
        "started": pl.Boolean,
        "finished": pl.Boolean,
    },
    # Club-level metadata. FPL leaves the attack/defence strength fields at 0
    # for most of a season and never populates played/won/lost at all, so the
    # projection derives its own ratings from player data and keeps these only
    # for reference.
    "fpl_teams": {
        "captured_at": TS,
        "team_id": pl.Int32,
        "name": pl.Utf8,
        "short_name": pl.Utf8,
        "strength": pl.Int32,
        "strength_overall_home": pl.Int32,
        "strength_overall_away": pl.Int32,
        "strength_attack_home": pl.Int32,
        "strength_attack_away": pl.Int32,
        "strength_defence_home": pl.Int32,
        "strength_defence_away": pl.Int32,
    },
    # The one table the ingest daemon does not write. BeManager scores from the
    # Sofascore match rating, which no FPL endpoint carries, so what a player
    # actually returned has to be entered by hand. That is a league's own
    # results rather than a second stats provider, which is why it does not
    # break the single-feed rule the rest of the app keeps.
    #
    # It is what makes the projection honest. Sofascore's rating is a
    # machine-learning model weighting each action by context, so no fixed
    # tariff built on FPL aggregates can reproduce it; the structural estimate
    # in league.py is a cold start, and these rows are what replace it.
    "league_scores": {
        "captured_at": TS,
        "gw": pl.Int32,
        "fpl_id": pl.Int32,
        "web_name": pl.Utf8,
        "rating": pl.Float64,  # the Sofascore rating, where it is known
        "points": pl.Int32,    # what the league actually paid
    },
}


def now() -> datetime:
    return datetime.now(UTC)


def empty(table: str) -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMAS[table])


def conform(table: str, rows: list[dict]) -> pl.DataFrame:
    """Build a DataFrame that exactly matches the declared schema."""
    schema = SCHEMAS[table]
    if not rows:
        return empty(table)
    df = pl.DataFrame(rows, schema_overrides=schema, strict=False)
    missing = [pl.lit(None).cast(dt).alias(c) for c, dt in schema.items() if c not in df.columns]
    if missing:
        df = df.with_columns(missing)
    return df.select([pl.col(c).cast(dt) for c, dt in schema.items()])


def write(table: str, df: pl.DataFrame) -> Path | None:
    """Append one parquet part file into today's partition."""
    if df.is_empty():
        return None
    part_dir = DATA_DIR / table / f"dt={now():%Y-%m-%d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    path = part_dir / f"part-{time.time_ns()}.parquet"
    df.write_parquet(path, compression="zstd", statistics=True)
    return path


def scan(table: str) -> pl.LazyFrame:
    """Lazy scan across every partition. Returns an empty frame if none exist.

    Adding a column to a schema has to stay non-breaking: files written before
    the change do not carry it, and the default scan raises on that mismatch
    rather than reading the older parts. `missing_columns="insert"` fills them
    with nulls, `extra_columns="ignore"` covers a column that has since been
    dropped, and the trailing select pins the column order to the schema.
    """
    root = DATA_DIR / table
    if not root.exists() or not any(root.rglob("*.parquet")):
        return empty(table).lazy()
    return pl.scan_parquet(
        root / "**/*.parquet",
        hive_partitioning=True,
        schema=SCHEMAS[table],
        missing_columns="insert",
        extra_columns="ignore",
    ).select(list(SCHEMAS[table]))


class Buffer:
    """Accumulates ticks in memory and flushes on an interval.

    One tick of a full matchday is a few hundred rows. Flushing each one would
    create thousands of files a day per table.
    """

    def __init__(self, table: str, flush_every: float):
        self.table = table
        self.flush_every = flush_every
        self._rows: list[dict] = []
        self._last = time.monotonic()

    def add(self, rows: list[dict]) -> None:
        self._rows.extend(rows)

    def maybe_flush(self, force: bool = False) -> int:
        if not self._rows:
            self._last = time.monotonic()
            return 0
        if not force and (time.monotonic() - self._last) < self.flush_every:
            return 0
        n = len(self._rows)
        write(self.table, conform(self.table, self._rows))
        self._rows.clear()
        self._last = time.monotonic()
        return n


def compact(table: str, keep_days: int = 2) -> None:
    """Merge each finished day's part files into a single file.

    Run this once a day. Scans stay fast and the file count stays sane.
    """
    root = DATA_DIR / table
    if not root.exists():
        return
    today = f"dt={now():%Y-%m-%d}"
    for part_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if part_dir.name >= today:
            continue  # never touch a partition the writer may still be appending to
        files = sorted(part_dir.glob("part-*.parquet"))
        if len(files) <= 1:
            continue
        merged = pl.read_parquet(files).unique()
        tmp = part_dir / "compacted.parquet.tmp"
        merged.write_parquet(tmp, compression="zstd", statistics=True)
        tmp.rename(part_dir / "compacted.parquet")
        for f in files:
            f.unlink()


def compact_cli() -> None:
    for table in SCHEMAS:
        compact(table)
        print(f"compacted {table}")
