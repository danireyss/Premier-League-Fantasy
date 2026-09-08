"""Live match analysis dashboard.

    uv run streamlit run app.py

This app only reads. The ingest daemon is a separate process — start it first
or every panel here will be empty.
"""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st

from flive import project, queries

st.set_page_config(page_title="Live match analysis", layout="wide")

REFRESH = 30


# Cache the read, not the render. A short TTL means several fragments in one
# pass share a single parquet scan instead of each triggering their own.
@st.cache_data(ttl=15, show_spinner=False)
def _live_fixtures() -> pl.DataFrame:
    return queries.live_fixtures()


@st.cache_data(ttl=15, show_spinner=False)
def _momentum(window: int) -> pl.DataFrame:
    return queries.player_momentum(window)


@st.cache_data(ttl=15, show_spinner=False)
def _timeline(fixture_id: int) -> pl.DataFrame:
    return queries.xg_timeline(fixture_id)


@st.cache_data(ttl=30, show_spinner=False)
def _fantasy(window: int) -> pl.DataFrame:
    return queries.fantasy_board(window)


@st.cache_data(ttl=300, show_spinner=False)
def _prices(hours: int) -> pl.DataFrame:
    return queries.price_watch(hours)


@st.cache_data(ttl=300, show_spinner=False)
def _players() -> pl.DataFrame:
    return queries.player_board()


@st.cache_data(ttl=300, show_spinner=False)
def _compare(min_minutes: int, per_90: bool) -> pl.DataFrame:
    return queries.compare_board(min_minutes, per_90)


@st.cache_data(ttl=300, show_spinner=False)
def _projections(gws: tuple[int, ...]) -> pl.DataFrame:
    return queries.projections(list(gws))


@st.cache_data(ttl=300, show_spinner=False)
def _gameweeks() -> list[int]:
    return queries.upcoming_gameweeks()


@st.cache_data(ttl=300, show_spinner=False)
def _strength() -> pl.DataFrame:
    return queries.team_strength()


@st.cache_data(ttl=300, show_spinner=False)
def _results(limit: int) -> pl.DataFrame:
    return queries.recent_results(limit)


@st.cache_data(ttl=300, show_spinner=False)
def _next_fixtures(limit: int) -> pl.DataFrame:
    return queries.next_fixtures(limit)


@st.cache_data(ttl=300, show_spinner=False)
def _next_kickoff():
    return queries.next_kickoff()


def _kickoff_note() -> str:
    """One line on when there will next be something live to show."""
    when = _next_kickoff()
    if when is None:
        return "No fixtures left to play."
    delta = when - queries.store.now()
    hours = delta.total_seconds() / 3600
    if hours < 0:
        return "The next fixture has kicked off; the daemon polls every minute."
    if hours < 24:
        return f"Next kickoff is in about {hours:.0f} hours ({when:%a %H:%M UTC})."
    return f"Next kickoff is {when:%a %d %b, %H:%M UTC} — in {hours / 24:.0f} days."


# Categorical slots 1-3 of the reference palette, which are the three that clear
# the all-pairs CVD and normal-vision floors a scatter needs. Goalkeepers are
# left off the attacking chart anyway — their xG is 0.00 — so three is enough.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
OUTFIELD = ["DEF", "MID", "FWD"]

# Slots 1-4. Grouped bars are read against their neighbours, so the adjacent
# pairlist applies and the fourth slot is available here even though the scatter
# above has to stop at three.
COMPARE_PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
MAX_COMPARE = 4

# The projection stack. Deductions take the red slot and everything else runs
# in slot order: red reads as the negative pole, and the deduction segment is
# the one that stacks left of zero. Validated on the pairs that actually touch
# on screen — deductions | 0 | appearance, attack, defence, bonus — which is
# not list order, since nothing ever renders yellow beside red.
STACK = {
    "light": {
        "Appearance": "#2a78d6", "Attack": "#eb6834", "Defence": "#1baf7a",
        "Bonus": "#eda100", "Deductions": "#e34948",
    },
    "dark": {
        "Appearance": "#3987e5", "Attack": "#d95926", "Defence": "#199e70",
        "Bonus": "#c98500", "Deductions": "#e66767",
    },
}
# How the nine scoring components collapse into the five the chart stacks. The
# table below it keeps all nine — past about seven classes adjacent colours
# blur, and the detail belongs in a table anyway.
STACK_OF = {
    "pts_appearance": "Appearance",
    "pts_goals": "Attack",
    "pts_assists": "Attack",
    "pts_clean_sheet": "Defence",
    "pts_defcon": "Defence",
    "pts_saves": "Defence",
    "pts_bonus": "Bonus",
    "pts_conceded": "Deductions",
    "pts_cards": "Deductions",
}
STACK_ORDER = ["Appearance", "Attack", "Defence", "Bonus", "Deductions"]


def _theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _labels(df: pl.DataFrame, n: int = 6) -> pl.DataFrame:
    """Pick the standouts to name, skipping any that would overprint another.

    Altair does no collision avoidance, and the players worth labelling cluster:
    two forwards on 2.3 xG and 0.1 xA land on the same pixel and render as
    illegible overstrike. Walk them by combined output and keep one only if it
    clears everything already kept, measured as a fraction of each axis range.
    """
    ranked = df.with_columns((pl.col("x") + pl.col("y")).alias("_r")).sort(
        "_r", descending=True
    )
    span_x = (df["x"].max() or 0) - (df["x"].min() or 0) or 1.0
    span_y = (df["y"].max() or 0) - (df["y"].min() or 0) or 1.0

    kept: list[dict] = []
    for row in ranked.iter_rows(named=True):
        if len(kept) >= n:
            break
        clear = all(
            abs(row["x"] - k["x"]) / span_x > 0.05
            or abs(row["y"] - k["y"]) / span_y > 0.07
            for k in kept
        )
        if clear:
            kept.append(row)
    return pl.DataFrame(kept, schema=ranked.schema) if kept else ranked.head(0)


def _slots(ids: list[int]) -> dict[int, int]:
    """Hold each player's colour slot for as long as he is selected.

    Colouring by position in the selection would repaint the survivors when one
    player is dropped, and a reader who has learned that Salah is the blue bar
    should not have to relearn it. Assign the lowest free slot on the way in,
    release it on the way out.
    """
    held: dict[int, int] = st.session_state.setdefault("compare_slots", {})
    for gone in [k for k in held if k not in ids]:
        del held[gone]
    for fpl_id in ids:
        if fpl_id not in held:
            held[fpl_id] = next(s for s in range(MAX_COMPARE) if s not in held.values())
    return held


with st.sidebar:
    st.subheader("Settings")
    window = st.slider("Momentum window (minutes)", 5, 45, 15, step=5)
    st.caption(
        "Momentum compares each player's current totals against their totals "
        f"{window} minutes ago."
    )
    st.divider()
    counts = queries.coverage()
    st.subheader("Stored snapshots")
    for table, n in counts.items():
        st.text(f"{n:>10,}  {table}")
    if not any(counts.values()):
        st.warning("No data yet. Start the ingest daemon with `uv run flive-ingest`.")

matches, players, compare, projection, fantasy, prices = st.tabs(
    ["Matches", "Players", "Compare", "Projection", "Fantasy", "Prices"]
)


with matches:

    @st.fragment(run_every=REFRESH)
    def live_panel() -> None:
        fx = _live_fixtures()
        if fx.is_empty():
            # The live loop writes only while a match is in progress, so
            # between gameweeks these tables are empty and stay that way for
            # days. The schedule is what there is to show, and it is not
            # nothing: results just played, and what is coming.
            st.info(
                "Nothing has been in play since the daemon started, so there "
                "are no live snapshots yet. " + _kickoff_note()
            )
            results, ahead = _results(10), _next_fixtures(10)
            if results.is_empty() and ahead.is_empty():
                st.caption(
                    "No fixture list either — the slow loop writes it on its "
                    "first tick."
                )
                return
            left, right = st.columns(2)
            with left:
                st.subheader("Latest results")
                if results.is_empty():
                    st.caption("Nothing played yet this season.")
                else:
                    st.dataframe(
                        results.select(
                            pl.col("gw").alias("GW"),
                            pl.col("home_name").alias("Home"),
                            pl.col("home_score").alias("H"),
                            pl.col("away_score").alias("A"),
                            pl.col("away_name").alias("Away"),
                        ),
                        hide_index=True,
                        width="stretch",
                    )
            with right:
                st.subheader("Coming up")
                if ahead.is_empty():
                    st.caption("No fixtures scheduled.")
                else:
                    st.dataframe(
                        ahead.select(
                            pl.col("gw").alias("GW"),
                            pl.col("kickoff_time").dt.strftime("%a %d %b %H:%M").alias("Kickoff"),
                            pl.col("home_name").alias("Home"),
                            pl.col("away_name").alias("Away"),
                            # Not "H diff" — this table sits beside one whose
                            # H and A columns hold scores, where "diff" reads
                            # as goal difference.
                            pl.col("home_difficulty").alias("Home FDR"),
                            pl.col("away_difficulty").alias("Away FDR"),
                        ),
                        hide_index=True,
                        width="stretch",
                    )
            st.caption(
                "**FDR** is FPL's fixture difficulty rating — how hard the "
                "match is for that side, 1 easiest to 5. The two differ "
                "because it accounts for venue. Expected points for these "
                "fixtures are on the Projection tab."
            )
            return

        in_play = fx.filter(pl.col("state").is_in(queries.LIVE_STATES))
        shown = in_play if not in_play.is_empty() else fx.head(10)

        if in_play.is_empty():
            st.caption("Nothing in play. Showing the ten most recent fixtures.")

        st.dataframe(
            shown.select(
                pl.col("home_name").alias("Home"),
                pl.col("home_score").alias("H"),
                pl.col("away_score").alias("A"),
                pl.col("away_name").alias("Away"),
                pl.col("minute").alias("Min"),
                pl.col("state").alias("State"),
                pl.col("home_xg").round(2).alias("Home xG"),
                pl.col("away_xg").round(2).alias("Away xG"),
                pl.col("xg_diff").round(2).alias("xG diff"),
            ),
            hide_index=True,
            width="stretch",
        )

        labels = {
            f"{r['home_name']} v {r['away_name']}": r["fixture_id"]
            for r in shown.iter_rows(named=True)
        }
        choice = st.selectbox("Inspect a fixture", list(labels), key="fixture_choice")
        fixture_id = labels[choice]

        left, right = st.columns([3, 2])

        with left:
            tl = _timeline(fixture_id)
            if tl.height < 2:
                st.caption("Not enough snapshots yet to draw a timeline.")
            else:
                long = tl.select("minute", "home_xg", "away_xg").unpivot(
                    index="minute", variable_name="side", value_name="xg"
                )
                # Step interpolation, not linear: xG only moves when a shot is
                # processed, so a smooth line would imply chances that never happened.
                chart = (
                    alt.Chart(long.to_pandas())
                    .mark_line(interpolate="step-after", strokeWidth=2)
                    .encode(
                        x=alt.X("minute:Q", title="Minute"),
                        y=alt.Y("xg:Q", title="Cumulative xG"),
                        color=alt.Color("side:N", title=None),
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, width="stretch")

        with right:
            mo = _momentum(window)
            if mo.is_empty():
                st.caption("No player snapshots in the window.")
            else:
                movers = mo.filter(pl.col("fixture_id") == fixture_id).head(10)
                if movers.is_empty():
                    st.caption("No players from this fixture in the window.")
                else:
                    st.dataframe(
                        movers.select(
                            pl.col("web_name").alias("Player"),
                            pl.col("d_xgi").round(3).alias(f"xGI +{window}m"),
                            pl.col("d_bps").alias("BPS"),
                            pl.col("xgi").round(2).alias("xGI total"),
                        ),
                        hide_index=True,
                        width="stretch",
                    )

    live_panel()


with players:
    pb = _players()
    if pb.is_empty():
        st.info(
            "No player data yet. The slow loop writes this on its first tick — "
            "start the ingest daemon with `uv run flive-ingest`."
        )
    else:
        # Filters in one row above the chart.
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4])
        with f1:
            pos = st.multiselect(
                "Position", ["GKP", "DEF", "MID", "FWD"], default=["DEF", "MID", "FWD"]
            )
        with f2:
            teams = st.multiselect(
                "Team", sorted(pb["team_name"].drop_nulls().unique().to_list())
            )
        with f3:
            max_min = int(pb["minutes"].max() or 0)
            min_minutes = st.slider("Minimum minutes", 0, max(max_min, 1), min(90, max_min))
        with f4:
            basis = st.radio("Basis", ["Totals", "Per 90"], horizontal=True)

        view = pb.filter(pl.col("minutes") >= min_minutes)
        if pos:
            view = view.filter(pl.col("position").is_in(pos))
        if teams:
            view = view.filter(pl.col("team_name").is_in(teams))

        search = st.text_input("Search player", placeholder="surname…")
        if search:
            view = view.filter(pl.col("web_name").str.to_lowercase().str.contains(search.lower()))

        if view.is_empty():
            st.warning("No players match those filters.")
        else:
            per90 = basis == "Per 90"
            sfx = " /90" if per90 else ""

            def stat(name: str) -> pl.Expr:
                """Per-90 columns exist for xG/xA/xGI from FPL and are derived
                for the ICT family; fall back to the total if one is absent."""
                col = f"{name}_per_90" if per90 else name
                return pl.col(col if col in view.columns else name)

            shown_n = min(view.height, 200)
            st.caption(
                f"{view.height} players match"
                + (f", showing the top {shown_n} by points" if view.height > shown_n else "")
                + " · **xG** expected goals · **xA** expected assists · "
                "**ICT** = Influence + Creativity + Threat"
            )

            table = view.select(
                pl.col("web_name").alias("Player"),
                pl.col("team_name").alias("Team"),
                pl.col("position").alias("Pos"),
                pl.col("price").round(1).alias("£"),
                pl.col("minutes").alias("Min"),
                pl.col("total_points").alias("Pts"),
                stat("xg").round(2).alias(f"xG{sfx}"),
                stat("xa").round(2).alias(f"xA{sfx}"),
                stat("xgi").round(2).alias(f"xGI{sfx}"),
                stat("ict_index").round(1).alias(f"ICT{sfx}"),
                stat("influence").round(1).alias(f"Infl{sfx}"),
                stat("creativity").round(1).alias(f"Creat{sfx}"),
                stat("threat").round(1).alias(f"Threat{sfx}"),
                pl.col("goals_scored").alias("G"),
                pl.col("assists").alias("A"),
                pl.col("selected_by_pct").alias("Owned %"),
                pl.col("points_per_game").alias("PPG"),
            )

            chart_data = view.filter(
                pl.col("position").is_in(OUTFIELD) & (pl.col("minutes") > 0)
            )
            if chart_data.height >= 2:
                xcol = "xg_per_90" if per90 else "xg"
                ycol = "xa_per_90" if per90 else "xa"
                chart_plot = chart_data.select(
                    pl.col("web_name").alias("player"),
                    pl.col("team_name").alias("team"),
                    pl.col("position").alias("pos"),
                    pl.col(xcol).alias("x"),
                    pl.col(ycol).alias("y"),
                    pl.col("minutes").alias("mins"),
                    pl.col("ict_index").alias("ict"),
                ).drop_nulls(["x", "y"])
                plot = chart_plot.to_pandas()

                colors = PALETTE[_theme()]
                ink = "#52514e" if _theme() == "light" else "#c3c2b7"
                ring = "#fcfcfb" if _theme() == "light" else "#1a1a19"

                # Recessive axes: faint grid, few ticks, no domain rule.
                ax = dict(
                    grid=True, gridOpacity=0.18, gridColor=ink, tickCount=7,
                    domain=False, ticks=False, labelColor=ink, titleColor=ink,
                    labelFontSize=11, titleFontSize=12, titlePadding=8,
                )
                # The surface ring keeps overlapping markers separable — the
                # bottom-left is dense by nature, since most players create little.
                pts = alt.Chart(plot).mark_circle(
                    size=110, opacity=0.8, stroke=ring, strokeWidth=1.5
                ).encode(
                    x=alt.X("x:Q", title=f"Expected goals{sfx}", axis=alt.Axis(**ax)),
                    y=alt.Y("y:Q", title=f"Expected assists{sfx}", axis=alt.Axis(**ax)),
                    color=alt.Color(
                        "pos:N",
                        title="Position",
                        scale=alt.Scale(domain=OUTFIELD, range=colors),
                        legend=alt.Legend(labelColor=ink, titleColor=ink, symbolStrokeWidth=0),
                    ),
                    tooltip=[
                        alt.Tooltip("player:N", title="Player"),
                        alt.Tooltip("team:N", title="Team"),
                        alt.Tooltip("pos:N", title="Pos"),
                        alt.Tooltip("x:Q", title=f"xG{sfx}", format=".2f"),
                        alt.Tooltip("y:Q", title=f"xA{sfx}", format=".2f"),
                        alt.Tooltip("ict:Q", title="ICT", format=".1f"),
                        alt.Tooltip("mins:Q", title="Minutes"),
                    ],
                )
                # Name only the standouts; every point labelled is unreadable.
                labels = (
                    alt.Chart(_labels(chart_plot).to_pandas())
                    .mark_text(align="left", dx=9, dy=-5, fontSize=11, color=ink)
                    .encode(x="x:Q", y="y:Q", text="player:N")
                )
                st.altair_chart(
                    (pts + labels)
                    .properties(height=380, padding={"right": 70, "top": 5})
                    .configure_view(strokeOpacity=0),
                    width="stretch",
                )

            st.dataframe(table.head(200), hide_index=True, width="stretch")
            st.caption(
                "FPL publishes no key-pass count — no pass, shot or chance "
                "counter exists in this API. **Creativity** is Opta's "
                "chance-creation index and **xA** the expected value of chances "
                "created; both stand in for it, and xA weights chance quality "
                "rather than merely counting."
            )


with compare:
    pool = _players()
    if pool.is_empty():
        st.info(
            "No player data yet. The slow loop writes this on its first tick — "
            "start the ingest daemon with `uv run flive-ingest`."
        )
    else:
        c1, c2, c3 = st.columns([5, 1.5, 2.5])
        with c2:
            cmp_basis = st.radio(
                "Basis", ["Totals", "Per 90"], horizontal=True, key="cmp_basis"
            )
        with c3:
            pool_max = int(pool["minutes"].max() or 0)
            pool_min = st.slider(
                "Peer pool: minimum minutes",
                0,
                max(pool_max, 1),
                min(90, pool_max),
                step=15,
                help=(
                    "Who a player is ranked against. Raising this drops cameo "
                    "appearances out of the pool, so percentiles compare him "
                    "against regulars rather than against everyone."
                ),
            )

        cmp_per90 = cmp_basis == "Per 90"
        board = _compare(pool_min, cmp_per90)
        options = {
            f"{r['web_name']} · {r['team_name']} ({r['position']})": r["fpl_id"]
            for r in board.iter_rows(named=True)
        }
        with c1:
            picked = st.multiselect(
                "Players",
                list(options),
                max_selections=MAX_COMPARE,
                placeholder="Pick two to four players…",
            )

        if len(picked) < 2:
            st.info("Pick at least two players to compare.")
        else:
            ids = [options[label] for label in picked]
            slots = _slots(ids)
            rows = {
                label: board.filter(pl.col("fpl_id") == options[label]).row(0, named=True)
                for label in picked
            }
            # Short names for the chart; the full label stays on the cards.
            short = {label: label.split(" · ")[0] for label in picked}
            order = list(short.values())

            theme = _theme()
            colors = [COMPARE_PALETTE[theme][slots[options[label]]] for label in picked]
            ink = "#52514e" if theme == "light" else "#c3c2b7"

            sfx = " /90" if cmp_per90 else ""

            cards = st.columns(len(picked))
            for col, label in zip(cards, picked):
                r = rows[label]
                with col:
                    st.markdown(
                        f"<span style='color:{colors[picked.index(label)]};"
                        f"font-size:1.6rem;line-height:0'>●</span> "
                        f"**{r['web_name']}**  \n"
                        f"{r['team_name']} · {r['position']} · £{r['price']:.1f}m",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{r['minutes']:,} min · {r['total_points']} pts · "
                        f"{r['selected_by_pct']:.1f}% owned"
                    )

            thin = [label for label in picked if rows[label]["minutes"] < pool_min]
            if thin:
                st.caption(
                    "Below the pool threshold, so unranked: "
                    + ", ".join(short[label] for label in thin)
                    + ". Their raw numbers are in the table; the bars leave them out."
                )

            # --- percentile profile -------------------------------------------
            plot = pl.DataFrame(
                [
                    {
                        "player": short[label],
                        "stat": stat,
                        "pct": rows[label][f"p_{queries.COMPARE_STATS[stat][0]}"],
                        "value": rows[label][f"v_{queries.COMPARE_STATS[stat][0]}"],
                    }
                    for stat in queries.PROFILE_STATS
                    for label in picked
                ],
                schema={"player": pl.Utf8, "stat": pl.Utf8, "pct": pl.Float64, "value": pl.Float64},
            ).drop_nulls("pct")

            if plot.is_empty():
                st.caption(
                    "None of these players clears the pool threshold, so there is "
                    "nothing to rank them against. Lower it, or read the table."
                )
            else:
                pdf = plot.to_pandas()
                # A fixed row height keeps the chart the same size whether two
                # players or four are on it; the bars thin out instead. Letting
                # Vega size the rows from the bar count collapsed the plot to
                # 90px for two players and it silently dropped every other
                # metric label.
                ranked_n = plot["player"].n_unique()
                row_h = 46
                bar_h = min(16, int(34 / ranked_n))
                ax = dict(
                    grid=True, gridOpacity=0.18, gridColor=ink, tickCount=5,
                    domain=False, ticks=False, labelColor=ink, titleColor=ink,
                    labelFontSize=11, titleFontSize=12, titlePadding=8,
                )
                scale = alt.Scale(domain=order, range=colors)
                base = alt.Chart(pdf).encode(
                    y=alt.Y(
                        "stat:N",
                        sort=queries.PROFILE_STATS,
                        title=None,
                        axis=alt.Axis(
                            domain=False, ticks=False, labelColor=ink,
                            labelFontSize=11, labelPadding=8, labelLimit=200,
                        ),
                    ),
                    yOffset=alt.YOffset("player:N", sort=order),
                    x=alt.X(
                        "pct:Q",
                        title="Percentile among positional peers",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(**ax),
                    ),
                )
                # An explicit bar height rather than a full offset band: the
                # slack becomes the surface gap that separates one player's bar
                # from the next, without drawing a border around either.
                bars = base.mark_bar(cornerRadiusEnd=4, height=bar_h).encode(
                    color=alt.Color(
                        "player:N",
                        title=None,
                        scale=scale,
                        sort=order,
                        legend=alt.Legend(
                            labelColor=ink, orient="top", symbolStrokeWidth=0, symbolType="square"
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("player:N", title="Player"),
                        alt.Tooltip("stat:N", title="Metric"),
                        alt.Tooltip("value:Q", title=f"Value{sfx}", format=".2f"),
                        alt.Tooltip("pct:Q", title="Percentile", format=".0f"),
                    ],
                )
                # Name the bars once, on the top group only. Identity then does
                # not rest on colour alone, and 28 labels do not fight the axis.
                names = (
                    base.transform_filter(alt.datum.stat == queries.PROFILE_STATS[0])
                    .mark_text(align="left", dx=6, fontSize=10, color=ink)
                    .encode(text="player:N")
                )
                st.altair_chart(
                    (bars + names)
                    .properties(
                        height=row_h * len(queries.PROFILE_STATS),
                        padding={"right": 80, "top": 5},
                    )
                    .configure_view(strokeOpacity=0),
                    width="stretch",
                )
                st.caption(
                    f"Percentiles are against players in the same position with "
                    f"{pool_min}+ minutes"
                    + (", on a per-90 basis" if cmp_per90 else "")
                    + ". ICT is omitted here — it is the sum of Influence, "
                    "Creativity and Threat, so it would count the same "
                    "performance a second time."
                )

            # --- the numbers ---------------------------------------------------
            table = {"Metric": [], **{short[label]: [] for label in picked}}
            if len(picked) == 2:
                table["Δ"] = []
            for stat, (column, scalable) in queries.COMPARE_STATS.items():
                table["Metric"].append(stat + (" /90" if cmp_per90 and scalable else ""))
                vals = [rows[label][f"v_{column}"] for label in picked]
                for label, value in zip(picked, vals):
                    table[short[label]].append(value)
                if len(picked) == 2:
                    both = vals[0] is not None and vals[1] is not None
                    table["Δ"].append(vals[0] - vals[1] if both else None)

            st.dataframe(
                pl.DataFrame(table).with_columns(
                    pl.col(pl.Float64).round(2)
                ),
                hide_index=True,
                width="stretch",
            )
            if len(picked) == 2:
                st.caption(f"Δ is {short[picked[0]]} minus {short[picked[1]]}.")


with projection:
    gw_options = _gameweeks()
    if not gw_options:
        st.info(
            "No fixture list yet. The slow loop writes it on its first tick — "
            "start the ingest daemon with `uv run flive-ingest`."
        )
    else:
        st.caption(
            "Expected points for fixtures not yet played, built from club "
            "attack and defence ratings, the opponent, the venue and how much "
            "of a match each player is likely to be on the pitch for. FPL "
            "publishes a projection of its own; this one does not look at it, "
            "so the two can be compared."
        )

        g1, g2, g3, g4 = st.columns([2.2, 2, 1.6, 1.6])
        with g1:
            gws = st.multiselect(
                "Gameweeks",
                gw_options,
                default=gw_options[:1],
                help=(
                    "Pick several to project a run of fixtures. Totals sum "
                    "every fixture a club has in the range, so a double "
                    "gameweek counts twice and a blank counts as nothing."
                ),
            )
        with g2:
            pos_pick = st.multiselect(
                "Position", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"]
            )
        with g3:
            max_price = st.number_input("Max price £m", 3.5, 20.0, 20.0, step=0.5)
        with g4:
            min_mins = st.number_input(
                "Min minutes played", 0, 3000, 90, step=90,
                help="Season minutes so far. Every rate in the model divides by "
                     "these, so a low bar lets in players whose numbers rest on "
                     "a cameo.",
            )

        if not gws:
            st.info("Pick at least one gameweek.")
        else:
            proj = _projections(tuple(sorted(gws)))
            if proj.is_empty():
                st.warning(
                    "Nothing to project yet. Club attack and defence ratings "
                    "are derived from matches already played, so there is "
                    "nothing to build on until the season's first gameweek "
                    "has finished."
                )
            else:
                view = proj.filter(
                    (pl.col("minutes") >= min_mins)
                    & (pl.col("price") <= max_price)
                    & (pl.col("xmins") > 0)
                )
                if pos_pick:
                    view = view.filter(pl.col("position").is_in(pos_pick))

                if view.is_empty():
                    st.warning("No players match those filters.")
                else:
                    # One row per player per fixture, so a run of gameweeks —
                    # and a double inside one — collapses by summing.
                    part_cols = list(project.COMPONENTS)
                    totals = (
                        view.group_by("fpl_id")
                        .agg(
                            pl.col("web_name").first(),
                            pl.col("team_name").first(),
                            pl.col("position").first(),
                            pl.col("price").first(),
                            pl.col("minutes").first(),
                            pl.col("xmins").mean().alias("xmins"),
                            pl.col("p60").first(),
                            pl.col("ep_next").first(),
                            pl.col("selected_by_pct").first(),
                            pl.col("news").first(),
                            pl.len().alias("fixtures"),
                            pl.col("difficulty").mean().alias("fdr"),
                            pl.concat_str(
                                [pl.col("opponent"), pl.col("venue")], separator=" ("
                            ).add(")").str.join(", ").alias("opponents"),
                            *[pl.col(c).sum() for c in part_cols],
                            pl.col("xp").sum().alias("xp"),
                        )
                        .with_columns((pl.col("xp") / pl.col("price")).alias("xp_per_m"))
                        .sort("xp", descending=True, nulls_last=True)
                    )

                    top_n = min(15, totals.height)
                    top = totals.head(top_n)

                    theme = _theme()
                    ink = "#52514e" if theme == "light" else "#c3c2b7"
                    ring = "#fcfcfb" if theme == "light" else "#1a1a19"

                    # Long form for the stack: five segments per player.
                    stacked = (
                        top.select("web_name", *part_cols)
                        .unpivot(index="web_name", variable_name="part", value_name="pts")
                        .with_columns(
                            pl.col("part")
                            .replace_strict(STACK_OF, return_dtype=pl.Utf8)
                            .alias("segment")
                        )
                        .group_by("web_name", "segment")
                        .agg(pl.col("pts").sum())
                        .filter(pl.col("pts").abs() > 0.005)
                        # Vega orders a stack alphabetically unless told
                        # otherwise, which would put Bonus before Defence and
                        # leave the segments in a different order per bar.
                        .with_columns(
                            pl.col("segment")
                            .replace_strict(
                                {name: i for i, name in enumerate(STACK_ORDER)},
                                return_dtype=pl.Int32,
                            )
                            .alias("rank")
                        )
                    )
                    order = top["web_name"].to_list()
                    colors = [STACK[theme][s] for s in STACK_ORDER]

                    ax = dict(
                        grid=True, gridOpacity=0.18, gridColor=ink, tickCount=6,
                        domain=False, ticks=False, labelColor=ink, titleColor=ink,
                        labelFontSize=11, titleFontSize=12, titlePadding=8,
                    )
                    bars = (
                        alt.Chart(stacked.to_pandas())
                        .mark_bar(cornerRadiusEnd=4, height=16, stroke=ring, strokeWidth=1)
                        .encode(
                            y=alt.Y(
                                "web_name:N", sort=order, title=None,
                                axis=alt.Axis(
                                    domain=False, ticks=False, labelColor=ink,
                                    labelFontSize=11, labelPadding=8, labelLimit=160,
                                ),
                            ),
                            x=alt.X(
                                "pts:Q",
                                title="Expected points"
                                + (f" over {len(gws)} gameweeks" if len(gws) > 1 else ""),
                                axis=alt.Axis(**ax),
                            ),
                            color=alt.Color(
                                "segment:N",
                                title=None,
                                scale=alt.Scale(domain=STACK_ORDER, range=colors),
                                sort=STACK_ORDER,
                                legend=alt.Legend(
                                    labelColor=ink, orient="top",
                                    symbolStrokeWidth=0, symbolType="square",
                                ),
                            ),
                            order=alt.Order("rank:Q", sort="ascending"),
                            tooltip=[
                                alt.Tooltip("web_name:N", title="Player"),
                                alt.Tooltip("segment:N", title="From"),
                                alt.Tooltip("pts:Q", title="Points", format=".2f"),
                            ],
                        )
                        .properties(
                            # Floored: Vega drops every other axis label when
                            # the plot is short, so a two-player result would
                            # silently lose a name.
                            height=max(120, 34 * top_n),
                            padding={"right": 20, "top": 5},
                        )
                        .configure_view(strokeOpacity=0)
                    )
                    st.altair_chart(bars, width="stretch")
                    st.caption(
                        f"Top {top_n} by expected points. The deductions segment "
                        "runs left of zero — goals conceded and cards are the "
                        "only components that subtract."
                    )

                    flagged = totals.filter(pl.col("news").is_not_null()).head(6)
                    if not flagged.is_empty():
                        with st.expander(f"{flagged.height} of these carry an FPL news flag"):
                            for r in flagged.iter_rows(named=True):
                                st.caption(f"**{r['web_name']}** — {r['news']}")

                    # FPL's own projection covers the next gameweek and nothing
                    # further, so it only belongs beside xP when the horizon is
                    # one gameweek too. Comparing it with a three-week total
                    # would read as this model being wildly optimistic.
                    single_gw = len(gws) == 1
                    st.dataframe(
                        totals.select(
                            pl.col("web_name").alias("Player"),
                            pl.col("team_name").alias("Team"),
                            pl.col("position").alias("Pos"),
                            pl.col("price").round(1).alias("£"),
                            pl.col("opponents").alias("Fixtures"),
                            pl.col("fdr").round(1).alias("FDR"),
                            pl.col("xp").round(2).alias("xP"),
                            pl.col("xp_per_m").round(2).alias("xP/£m"),
                            *(
                                [pl.col("ep_next").alias("FPL ep")]
                                if single_gw
                                else []
                            ),
                            pl.col("xmins").round(0).alias("xMins"),
                            (pl.col("p60") * 100).round(0).alias("P(60) %"),
                            pl.col("pts_appearance").round(2).alias("Appear"),
                            pl.col("pts_goals").round(2).alias("Goals"),
                            pl.col("pts_assists").round(2).alias("Assists"),
                            pl.col("pts_clean_sheet").round(2).alias("CS"),
                            pl.col("pts_defcon").round(2).alias("DefCon"),
                            pl.col("pts_saves").round(2).alias("Saves"),
                            pl.col("pts_bonus").round(2).alias("Bonus"),
                            pl.col("pts_conceded").round(2).alias("Conceded"),
                            pl.col("pts_cards").round(2).alias("Cards"),
                            pl.col("selected_by_pct").alias("Owned %"),
                        ).head(200),
                        hide_index=True,
                        width="stretch",
                    )
                    st.caption(
                        "**xP** is this model"
                        + (
                            "; **FPL ep** is FPL's own figure, shown as a "
                            "cross-check rather than an input"
                            if single_gw
                            else ", summed over every fixture in range"
                        )
                        + ". **FDR** is FPL's fixture difficulty, 1 easiest to "
                        "5, averaged over those fixtures — displayed only, never "
                        "used in the arithmetic, since the model derives its own "
                        "club ratings. **xMins** is per match."
                    )

                    with st.expander("Club ratings behind these numbers"):
                        strength = _strength()
                        if strength.is_empty():
                            st.caption("Not enough finished fixtures yet.")
                        else:
                            st.caption(
                                "Attack and defence are league-relative: 1.00 is "
                                "average, higher attack is better, higher defence "
                                "is leakier. Both are shrunk toward 1.00 by how "
                                "few matches they rest on, which is why they "
                                "cluster early in a season."
                            )
                            st.dataframe(
                                strength.select(
                                    pl.col("team_name").alias("Team"),
                                    pl.col("matches").alias("Played"),
                                    pl.col("xg_pm").round(2).alias("xG / match"),
                                    pl.col("xgc_pm").round(2).alias("xGC / match"),
                                    pl.col("attack").round(3).alias("Attack"),
                                    pl.col("defence").round(3).alias("Defence"),
                                ),
                                hide_index=True,
                                width="stretch",
                            )


with fantasy:

    @st.fragment(run_every=REFRESH * 2)
    def fantasy_panel() -> None:
        board = _fantasy(window)
        if board.is_empty():
            # This board is live gameweek scoring, so unlike the Matches tab
            # there is no standing-in for it — season totals are a different
            # question and the Players tab already answers it. Say when it
            # will fill and point at what does work now.
            st.info(
                "This board shows live gameweek scoring, so it fills once a "
                "gameweek is under way. " + _kickoff_note()
            )
            st.caption(
                "In the meantime: **Players** has season totals, **Projection** "
                "has expected points for the fixtures ahead."
            )
            return

        if "bonus_final" in board.columns and not board["bonus_final"].fill_null(False).any():
            st.warning(
                "Bonus points are provisional. BPS moves during matches and "
                "only settles afterwards, so these totals will change."
            )

        cols = [
            pl.col("web_name").alias("Player"),
            pl.col("team_name").alias("Team"),
            pl.col("position").alias("Pos"),
            pl.col("price").round(1).alias("Price"),
            pl.col("total_points").alias("Pts"),
            pl.col("bps").alias("BPS"),
            pl.col("minutes").alias("Min"),
            pl.col("xgi").round(2).alias("xGI"),
            pl.col("selected_by_pct").alias("Owned %"),
        ]
        if "d_xgi" in board.columns:
            cols.insert(-1, pl.col("d_xgi").round(3).alias(f"xGI +{window}m"))

        st.dataframe(
            board.filter(pl.col("minutes") > 0).select(cols).head(50),
            hide_index=True,
            width="stretch",
        )

    fantasy_panel()


with prices:
    hours = st.select_slider("Look back", options=[6, 12, 24, 48, 168], value=24)
    pw = _prices(hours)
    if pw.is_empty():
        st.info(
            "No movement recorded. Prices and ownership come from the slow loop, "
            "which runs every six hours."
        )
    else:
        st.dataframe(
            pw.select(
                pl.col("web_name").alias("Player"),
                pl.col("team_name").alias("Team"),
                pl.col("position").alias("Pos"),
                pl.col("price").round(1).alias("Price"),
                (pl.col("cost_delta") / 10).round(1).alias("Price change"),
                pl.col("ownership").round(1).alias("Owned %"),
                pl.col("ownership_delta").round(2).alias("Ownership change"),
            ).head(50),
            hide_index=True,
            width="stretch",
        )
