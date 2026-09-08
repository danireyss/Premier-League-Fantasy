# flive

A Premier League dashboard built on the FPL API alone — no API key, no paid
feed, no second provider. Polars for the analytics, Streamlit for the UI,
parquet for the store, uv for everything else.

Three things it does that FPL itself does not: it keeps a history of the live
feed FPL forgets, it ranks players against their positional peers rather than
against everyone, and it projects expected points for fixtures that have not
been played yet.

## What it shows

Six tabs.

**Matches** — live scores, team xG, and a cumulative-xG timeline for a chosen
fixture, beside the players gaining the most in the last *n* minutes. Between
gameweeks it shows the schedule instead: recent results, and what is coming
with kickoff times and difficulty.

**Players** — season totals for everyone, filterable by position, club and
minutes, on totals or per-90, with an xG-against-xA scatter.

**Compare** — two to four players side by side, each metric as a percentile
against positional peers, plus raw numbers and a delta column for exactly two.

**Projection** — expected points for fixtures not yet played, broken into the
components that make them up:

```
xP = appearance + goals + assists + clean sheet + defensive contribution
     + saves + bonus − goals conceded − cards
```

Filterable by gameweek, position, club, price, minutes and name. Club attack
and defence ratings are derived from matches already played, so it has nothing
to show until the season's first gameweek has finished.

**Fantasy** — live gameweek scoring beside the performance driving it. Fills
once a gameweek is under way.

**Prices** — players whose price or ownership has moved, over a chosen window.

## How to start it

```bash
uv sync
```

Two processes, always — one writes, one reads:

```bash
uv run flive-ingest                    # terminal 1 — the daemon, writes
uv run streamlit run app.py            # terminal 2 — the dashboard, reads
```

The app only ever reads. **Start the daemon first**, or every panel will be
empty. Data lands in `./data`; set `FLIVE_DATA_DIR` to put it elsewhere.

Both at once, dashboard on :8501:

```bash
docker compose up --build
```

And once a day, ideally from cron:

```bash
uv run flive-compact
```

## Possible improvements

**Last season as a prior.** Every rate in the projection is season-to-date, so
in August it rests on two or three matches. FPL's `/element-summary/{id}/`
endpoint returns `history_past` — minutes, goals, assists, xG, xA and xGC per
prior season — which would let a player's current rate be shrunk toward his
last-season per-90 the way club ratings are already shrunk toward the league
average. Note that xG only goes back to 2022/23 there, `defensive_contribution`
only to 2024/25, and players new to the league have no history at all.

**Club ratings from your own history.** The same problem one level up, and
harder: `history_past` carries no team field, and FPL serves no past-season
fixtures. Since FBref lost its Opta licence in January 2026 there is no free
source for it either. The cheap fix is patience — the slow loop already
snapshots everything to parquet, so running the daemon to season's end gives
you last season's club ratings next August from data you already own.

**Set-piece takers.** `penalties_order`, `corners_order` and `freekicks_order`
are already ingested and stored, and nothing reads them. A club's penalty taker
has a materially different goal rate to his xG per 90 alone suggests, and it is
the largest signal currently sitting unused in the store.

**A squad optimiser.** The projection ranks players individually; picking a
best XI under FPL's budget, formation and three-per-club rules is the obvious
next step, and is a straightforward knapsack over the numbers already produced.

**Sharper scoring rules.** Goals conceded is modelled linearly where FPL's real
rule steps every second goal, and own goals and missed penalties are not
modelled at all.

**Fixture-level history.** Player rates are season aggregates, so there is no
home/away or opponent-adjusted split. `player_snapshots` accumulates the raw
material for it a gameweek at a time.

---

[NOTES.md](NOTES.md) has the design rationale, the projection's derivations,
and the things that will bite you if you change the ingest or the schemas.
