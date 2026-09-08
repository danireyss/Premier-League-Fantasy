# flive

Live Premier League match analysis and fantasy tracking. Polars for the
analytics, Streamlit for the UI, parquet for the store, uv for everything else.

## Setup

```bash
uv sync
export FLIVE_DATA_DIR=./data       # optional, defaults to ./data
```

No API key. FPL is the only provider and it is open.

Two processes, always:

```bash
uv run flive-ingest                    # terminal 1 — writes
uv run streamlit run app.py            # terminal 2 — reads
```

Or both at once:

```bash
docker compose up --build              # dashboard on :8501
```

Once a day, ideally from cron:

```bash
uv run flive-compact
```

## Why it is shaped like this

**One provider, one keyspace.** This app covers the Premier League and nothing
else, and the FPL API covers the Premier League completely — points, price,
ownership, BPS, and since Opta data landed in it, xG, xA and the defensive
numbers too. Everything is keyed by `fpl_id`, so there is no cross-provider id
mapping to build, verify, or re-run after a transfer window.

A paid feed buys live match-clock granularity — shots, touches, key passes,
per-minute state — for roughly €58/mo once the xG add-on is included. That is
the only thing missing here, and momentum does not need it: the time series is
built from our own snapshots, not the provider's clock.

**The poller does not live inside Streamlit.** Streamlit reruns the entire
script on every widget interaction and every autorefresh. A background thread
started at module scope gets duplicated across sessions, restarted constantly,
and will quietly multiply your request volume by the number of open browser
tabs. The daemon writes, the app reads, and they never share a process.

**Parquet, not DuckDB.** DuckDB holds an exclusive lock on its file for a
writing process. With the daemon and the app in separate processes, a shared
`.duckdb` file means Streamlit either can't open it or blocks the writer.
Hive-partitioned parquet gives one writer and unlimited concurrent readers for
free, and `pl.scan_parquet` pushes filters and projections into the files.

**Nothing is ever updated.** Every tick appends a new snapshot. This is the
whole point: FPL tells you current state and keeps no history of its own live
feed. The time series is yours to build, and it is what makes
`player_momentum()` possible — FPL will tell you a player has 0.61 xGI, but not
that 0.4 of it arrived in the last twelve minutes.

**Two loops, not one.**

| Loop  | Interval | Source                     |
|-------|----------|----------------------------|
| live  | 60s      | `/fixtures/` + `/event/{gw}/live/` |
| slow  | 6h       | `/bootstrap-static/`       |

The live loop drops to a 5-minute idle poll when nothing is in play. Polling an
unsupported API every minute overnight, against data that will not change for
hours, is the fastest way to get your IP throttled.

## Layout

```
src/flive/
  config.py    tunables, cadences, stat field maps
  store.py     parquet schemas, buffered writes, lazy scans, compaction
  clients.py   FPL HTTP, backoff
  parse.py     provider JSON -> flat rows
  ingest.py    the two loops
  queries.py   Polars analytics
app.py         Streamlit dashboard
```

## Things that will bite you

**FPL sends numbers as strings.** `expected_goals` is `"0.61"`, and an unplayed
player's is `""`, not null. `parse._f` coerces and returns `None` on both. Do
not trust a field's type because it looked numeric once.

**Team xG is summed from players.** FPL publishes no team-level xG, so
`team_xg_from_players` adds up each side's live per-player figures. It tracks
the real number closely but is not identical to it — own goals and unattributed
chances sit outside it.

**FPL bonus points are provisional.** BPS shifts throughout a match and only
settles afterwards. The daemon polls `/event-status/` and stores `bonus_final`
alongside every row; the app warns when it's false. Skip this and you'll show
users totals that change underneath them.

**Momentum must not cross a gameweek boundary.** Totals reset to zero at the
rollover, so a window straddling one would read the reset as a large negative
move. `player_momentum` groups by `gw` as well as `fpl_id` to prevent it.

**Double gameweeks give a player two fixtures.** The live stat line is
aggregated across both, but a row carries one `fixture_id` — the most recent,
from `explain[-1]`. Per-fixture splits are not available from this endpoint.

**xG is stepped, not continuous.** It only moves when a shot is registered and
processed. The chart uses `step-after` interpolation deliberately — a smooth
line implies chances that did not happen.

**FPL blocks default user agents.** `config.FPL_HEADERS` sets a browser one.
These endpoints are undocumented and unsupported — cache hard, don't hammer,
and don't put them on a critical path you can't degrade gracefully.

**Match state is derived, not given.** FPL has no state enum, only `started` /
`finished` / `finished_provisional` flags. `parse.fixture_state` turns them into
`UPCOMING` / `LIVE` / `PROV` / `FT`. There is no half-time signal.
