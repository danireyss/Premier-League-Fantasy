"""Live match analysis dashboard.

    uv run streamlit run app.py

This app only reads. The ingest daemon is a separate process — start it first
or every panel here will be empty.
"""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st

from flive import queries

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


# Categorical slots 1-3 of the reference palette, which are the three that clear
# the all-pairs CVD and normal-vision floors a scatter needs. Goalkeepers are
# left off the attacking chart anyway — their xG is 0.00 — so three is enough.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
OUTFIELD = ["DEF", "MID", "FWD"]


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

matches, players, fantasy, prices = st.tabs(["Matches", "Players", "Fantasy", "Prices"])


with matches:

    @st.fragment(run_every=REFRESH)
    def live_panel() -> None:
        fx = _live_fixtures()
        if fx.is_empty():
            st.info("No fixtures recorded yet. The daemon idles until kickoff.")
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


with fantasy:

    @st.fragment(run_every=REFRESH * 2)
    def fantasy_panel() -> None:
        board = _fantasy(window)
        if board.is_empty():
            st.info(
                "No gameweek data yet. The live loop populates this once a "
                "gameweek is active."
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
