# flive

Premier League match analysis and fantasy tracking, built on the FPL API alone.
Polars for the analytics, Streamlit for the UI, parquet for the store, uv for
everything else.

It does three things beyond reporting what FPL already tells you: it builds a
time series out of snapshots FPL keeps no history of, it ranks players against
their positional peers rather than against everyone, and it projects expected
points for fixtures that have not been played.

## Setup

```bash
uv sync
export FLIVE_DATA_DIR=./data       # optional, defaults to ./data
```

No API key. FPL is the only provider and it is open.

Two processes, always — one writes, one reads:

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

The app only ever reads. Start the daemon first or every panel will be empty.

## What's in it

**Matches** — live scores, team xG and a stepped cumulative-xG timeline for a
chosen fixture, beside the players gaining the most in the last *n* minutes.
Between gameweeks there is no live data, so it shows the schedule instead:
results just played, and what is coming with kickoff times and difficulty.

**Players** — season totals for everyone, filterable by position, club and
minutes, on totals or per-90, with an xG-against-xA scatter that names only the
standouts.

**Compare** — two to four players side by side, each metric shown as a
percentile against positional peers, plus the raw numbers and a delta column
when exactly two are picked.

**Projection** — expected points for fixtures not yet played, broken into the
nine components that make them up, over one gameweek or a run of them.

**Fantasy** — live gameweek scoring beside the underlying performance driving
it. Fills once a gameweek is under way.

**Prices** — players whose price or ownership has moved, over a chosen window.

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

**Comparisons are percentiles, not raw numbers.** A defender's 0.4 xA and a
striker's are not the same achievement, so the Compare tab ranks each player
against others in the same position who clear a minutes threshold. Without that
threshold, one bright cameo outranks a season of steady work on any per-90
figure — so the pool is the control, and it is a visible one.

**Two loops, not one.**

| Loop  | Interval | Source                             | Writes                                  |
|-------|----------|------------------------------------|-----------------------------------------|
| live  | 60s      | `/fixtures/` + `/event/{gw}/live/` | `fixture_snapshots`, `player_snapshots` |
| slow  | 6h       | `/bootstrap-static/`, `/fixtures/` | `fpl_players`, `fpl_teams`, `fixtures`  |

The live loop drops to a 5-minute idle poll when nothing is in play. Polling an
unsupported API every minute overnight, against data that will not change for
hours, is the fastest way to get your IP throttled.

A consequence worth knowing: the two snapshot tables are written **only while a
match is in progress**. Between one gameweek and the next — four days, usually
— they are empty, and the Matches and Fantasy tabs have no live data to draw.
Matches falls back to the schedule; Fantasy is live gameweek scoring with no
standing-in for it, so it says when it will fill and points at the tabs that do
work. Neither is a fault, and both say so rather than showing an empty panel.

## The projection

```
xP = appearance + goals + assists + clean sheet + defensive contribution
     + saves + bonus − goals conceded − cards
```

Every scoring rule is a named constant in `config.py`, and each component is
returned as its own column, so a number can always be taken apart. `project.py`
is the only module in the app that guesses; it keeps its assumptions in the
open.

**Club ratings are derived, not taken.** FPL ships `strength_attack_home` and
its three siblings as `0` for most of a season and never fills in the team
table's played/won/lost at all, so neither can be built on. Two identities in
the player data replace them:

* A club's xG is exactly the sum of its players' xG — every chance belongs to
  whoever took it.
* Each player's `xgc` is what his team conceded while he was on the pitch, so
  the squad's summed `xgc` counts every conceded chance eleven times, once per
  player out there for it. Divide by eleven for the club's own figure.

Both are checked against clubs whose first-choice keeper has played every
minute, where his `xgc` must equal the club total. It does — and where two
keepers split the season, the sum-over-eleven is right and the single-keeper
reading is not.

Ratings are then shrunk toward the league average in proportion to matches
played, because three games in, one thrashing otherwise decides a rating for a
month. Venue is applied once, as a ±10% factor on the attacking side. FPL's
fixture difficulty is shown next to the numbers but never enters them: it is
already venue-aware, so using both would count home advantage twice.

**Minutes are modelled per club match**, not per appearance, so a player who
has missed half his side's games reads as such. Appearance points split into
P(plays) and P(reaches 60), since a regular substitute earns one and never the
other. `chance_of_playing_next_round` overrides the status flag whenever FPL
publishes one.

FPL's own `ep_next` is displayed as a cross-check but is never an input. Across
players with 90+ minutes the two correlate at **r ≈ 0.80** with similar means
(2.9 vs 3.2), which is about right: built from the same reality, not from each
other.

**What not to trust.** Rates are season-to-date, so in August they rest on a
handful of matches — shrinkage rescues the club ratings, nothing rescues a
player rate built from 90 minutes. Read xP next to the minutes column. The
`−1 per 2 conceded` rule is modelled linearly where the real rule steps every
second goal; it errs the same way for everyone.

## Layout

```
src/flive/
  config.py    tunables, cadences, stat field maps, scoring constants
  store.py     parquet schemas, buffered writes, lazy scans, compaction
  clients.py   FPL HTTP, backoff
  parse.py     provider JSON -> flat rows
  ingest.py    the two loops
  queries.py   Polars analytics
  project.py   expected points for fixtures not yet played
app.py         Streamlit dashboard
```

Five tables. `fixture_snapshots` and `player_snapshots` are append-only time
series from the live loop. `fpl_players`, `fpl_teams` and `fixtures` are
rewritten whole by the slow loop — the last is the season's fixture list rather
than match state, which is what the projection needs and what the Matches tab
falls back to.

## Working on it in Docker

`src/` and `app.py` are bind-mounted into both containers, so they follow your
working tree — edit a file and the dashboard reruns on its own. Two things do
not follow along:

* **The ingest daemon.** A plain process with no watcher, so changes to
  `ingest.py`, `parse.py` or `store.py` need `docker compose restart ingest`.
* **Dependencies.** `pyproject.toml` is baked in at build time; adding one
  means `docker compose up -d --build`.

The Streamlit watcher is pinned to `poll`. A bind mount does not deliver
inotify events into a container, so the default `auto` watcher sits silent and
the app serves stale code from a mount that is perfectly up to date — with
nothing in the logs to say so.

Without the mounts the same trap is worse: the image holds a copy of the source
from whenever it was last built, while `./data` is live. Old code reading newly
written data is how a schema change surfaces as a `SchemaError` at runtime.

## Things that will bite you

**FPL sends numbers as strings.** `expected_goals` is `"0.61"`, and an unplayed
player's is `""`, not null. `parse._f` coerces and returns `None` on both. Do
not trust a field's type because it looked numeric once.

**Adding a column to a schema is not free.** Files written before the change do
not carry it, and a plain `scan_parquet` across the glob raises on the
mismatch. `store.scan` pins the declared schema and passes
`missing_columns="insert"`, so old parts read back with nulls in the new
columns. Do not reach for `collect_schema()` to work out what is there: on a
multi-file scan it resolves to the first file's schema, which silently drops
the new column from every file that does have it — and the failure is not an
error but a column of zeros.

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
The projection sidesteps this by keying on fixture rather than gameweek, so a
double is two rows that sum and a blank is none.

**xG is stepped, not continuous.** It only moves when a shot is registered and
processed. The chart uses `step-after` interpolation deliberately — a smooth
line implies chances that did not happen.

**FPL blocks default user agents.** `config.FPL_HEADERS` sets a browser one.
These endpoints are undocumented and unsupported — cache hard, don't hammer,
and don't put them on a critical path you can't degrade gracefully.

**Match state is derived, not given.** FPL has no state enum, only `started` /
`finished` / `finished_provisional` flags. `parse.fixture_state` turns them into
`UPCOMING` / `LIVE` / `PROV` / `FT`. There is no half-time signal.
