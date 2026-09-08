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

matches, fantasy, prices = st.tabs(["Matches", "Fantasy", "Prices"])


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
            use_container_width=True,
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
                st.altair_chart(chart, use_container_width=True)

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
                        use_container_width=True,
                    )

    live_panel()


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
            use_container_width=True,
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
            use_container_width=True,
        )
