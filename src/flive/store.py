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
    """Lazy scan across every partition. Returns an empty frame if none exist."""
    root = DATA_DIR / table
    if not root.exists() or not any(root.rglob("*.parquet")):
        return empty(table).lazy()
    return pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True).select(
        list(SCHEMAS[table].keys())
    )


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
