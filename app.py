"""Live match analysis dashboard.

    uv run streamlit run app.py

This app only reads. The ingest daemon is a separate process — start it first
or every panel here will be empty.
"""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st

from flive import league, queries

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


# Both boards below are shaped by queries.COMPARE_STATS and POSITION_PROFILES,
# and Streamlit keys a cache entry on the wrapper's own source and arguments —
# never on a module constant the wrapper happens to read. Adding a metric
# therefore changed the board's columns without changing anything the cache
# could see, and a board cached under the previous metric set was handed back
# to a profile that then asked it for a percentile column it had never been
# built with. Passing the metric set in as an argument is what makes that
# dependency visible to the cache.
METRIC_KEY = ",".join(queries.COMPARE_STATS) + "|" + ",".join(
    f"{pos}:{','.join(profile)}"
    for pos, profile in sorted(queries.POSITION_PROFILES.items())
)


@st.cache_data(ttl=300, show_spinner=False)
def _compare(min_minutes: int, per_90: bool, metrics: str = METRIC_KEY) -> pl.DataFrame:
    return queries.compare_board(min_minutes, per_90)


@st.cache_data(ttl=300, show_spinner=False)
def _pos_scores(
    position: str, min_minutes: int, per_90: bool, metrics: str = METRIC_KEY
) -> pl.DataFrame:
    return queries.position_scores(position, min_minutes, per_90)


def _name_match(term: str) -> pl.Expr:
    """Search both the display name and the full one.

    FPL's `web_name` is the surname alone for most players, so a search for
    "geovany quenda" — or any first name — misses a player who is plainly
    there. `full_name` carries both halves.
    """
    needle = term.strip().lower()
    return pl.any_horizontal(
        pl.col(c).fill_null("").str.to_lowercase().str.contains(needle, literal=True)
        for c in ("web_name", "full_name")
    )


def _excluded_note(named: pl.DataFrame, min_mins: int, max_price: float,
                   pos_pick: list[str], team_pick: list[str]) -> str:
    """Why the players a search found were then filtered away.

    Worth spelling out: the default 90-minute bar hides every squad player who
    has only come off the bench, and "no players match those filters" gives no
    hint that the player exists and it was the bar that removed him.
    """
    reasons = []
    for r in named.unique(subset=["fpl_id"], keep="first").head(4).iter_rows(named=True):
        if r["minutes"] < min_mins:
            why = f"{r['minutes']} minutes played, under the {min_mins} bar"
        elif r["price"] > max_price:
            why = f"£{r['price']:.1f}m, over the £{max_price:.1f}m cap"
        elif pos_pick and r["position"] not in pos_pick:
            why = f"a {r['position']}, not in the position filter"
        elif team_pick and r["team_name"] not in team_pick:
            why = f"at {r['team_name']}, not in the team filter"
        else:
            why = "not expected on the pitch for these fixtures"
        reasons.append(f"**{r['web_name']}** ({r['team_name']}) — {why}")
    found = named["fpl_id"].n_unique()
    more = f" …and {found - 4} more." if found > 4 else ""
    return "Found, but filtered out: " + "; ".join(reasons) + "." + more


@st.cache_data(ttl=300, show_spinner=False)
def _league_projections(gws: tuple[int, ...]) -> pl.DataFrame:
    return queries.league_projections(list(gws))


@st.cache_data(ttl=60, show_spinner=False)
def _observed_ratings() -> pl.DataFrame:
    return league.observed_ratings()


@st.cache_data(ttl=60, show_spinner=False)
def _calibration() -> pl.DataFrame:
    return league.calibration()


def _resolve_returns(edited, gw: int) -> tuple[list[dict], list[str]]:
    """Turn edited rows into `league_scores` rows, reporting names that missed.

    Matching is on web_name, case-folded, falling back to a unique substring:
    the editor takes free text so that entering a gameweek is not fifteen trips
    through a six-hundred-name dropdown, and the cost of that is saying plainly
    when a name found nobody rather than dropping it silently.
    """
    board = _players()
    if board.is_empty():
        return [], []
    frame = edited if isinstance(edited, pl.DataFrame) else pl.from_pandas(edited)
    lookup = {
        name.casefold(): (fid, name)
        for fid, name in zip(board["fpl_id"].to_list(), board["web_name"].to_list())
    }
    rows: list[dict] = []
    unmatched: list[str] = []
    for r in frame.iter_rows(named=True):
        typed = (r.get("player") or "").strip()
        if not typed or (r.get("points") is None and r.get("rating") is None):
            continue
        hit = lookup.get(typed.casefold())
        if hit is None:
            partial = [v for k, v in lookup.items() if typed.casefold() in k]
            if len(partial) != 1:
                unmatched.append(typed)
                continue
            hit = partial[0]
        rows.append({
            "gw": gw, "fpl_id": hit[0], "web_name": hit[1],
            "points": int(r["points"]) if r.get("points") is not None else None,
            "rating": float(r["rating"]) if r.get("rating") is not None else None,
        })
    return rows, unmatched


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

# The midfield matrix. A percentile is a position either side of the median
# midfielder, so the scale is diverging rather than sequential: neutral at the
# 50th, both poles saturating outward. Blue-red is the documented diverging
# pair — the two poles read as opposite and the grey midpoint reads as nothing
# to see, which blue-aqua does not.
#
# Each arm has to hold one hue, which is what decides the stops. Light blends
# toward near-white, and that holds hue on its own, so three stops is the whole
# ramp. Dark cannot: its midpoint is near-black, and running a bright pole
# straight into it desaturates and darkens at the same time — rendered, the
# blue arm came out teal and the red arm brown, both arms reading as a hue
# nobody put there. Each dark arm therefore gets an explicit on-hue step, so
# only lightness moves along it and the hue stays at 0° and 213°.
DIVERGING = {
    "light": ([0, 50, 100], ["#e34948", "#f0efec", "#2a78d6"]),
    "dark": (
        [0, 25, 50, 75, 100],
        ["#e66767", "#9c4242", "#383835", "#1c5cab", "#3987e5"],
    ),
}
# Where the in-cell number stops being legible against the cell under it. Every
# stop on the light ramp holds at least 4.46:1 against primary ink, so light
# keeps one colour throughout. Dark has to flip at both poles, where the cell
# is at full brightness: white across the middle, dark ink at the ends.
DARK_INK_FLIP = (12, 92)
MATRIX_ROWS = 12
RANK_ROWS = 15

# Slots 1-4. Grouped bars are read against their neighbours, so the adjacent
# pairlist applies and the fourth slot is available here even though the scatter
# above has to stop at three.
COMPARE_PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
MAX_COMPARE = 4

# The BeManager stack. Three components, not nine: the rating is one number
# that already contains everything a player did, and only the goal and assist
# bonuses sit outside it. Slots 1-3 of the categorical order, which is the run
# that stays separable for a colourblind reader.
LEAGUE_STACK = {
    "light": {"Rating": "#2a78d6", "Goal bonus": "#eb6834", "Assist bonus": "#1baf7a"},
    "dark": {"Rating": "#3987e5", "Goal bonus": "#d95926", "Assist bonus": "#199e70"},
}
LEAGUE_STACK_ORDER = ["Rating", "Goal bonus", "Assist bonus"]
# Ceiling and floor are worth ranking on directly in a league whose table pays
# down to -4: one format pays for hauls, this one also charges for bad games.
LEAGUE_RANKS = ["Total", "Rating", "Ceiling", "Floor"]
LEAGUE_RANK_COL = {
    "Total": "xp", "Rating": "rating_mu",
    "Ceiling": "p_excellent", "Floor": "p_negative",
}


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


# What each position's profile is called on screen, and the notes that belong
# under it. The absences are stated on the chart rather than left for a reader
# to assume a proxy is the real thing.
PROFILE_LABEL = {"MID": "midfielder", "DEF": "defender"}
PROFILE_NOTE = {
    "MID": (
        "**Creativity** is Opta's chance-creation index, the closest thing here "
        "to key passes and big chances created; interceptions are inside "
        "**clearances, blocks & int.**, which FPL never splits. Touches, pass "
        "accuracy, long balls and duels won are absent from the FPL API "
        "entirely, so they are not on this chart — nothing published here would "
        "stand in for them honestly."
    ),
    "DEF": (
        "**Clearances, blocks & int.** is one column because FPL sums those "
        "three and never splits them. **Creativity** and **xA** stand in for key "
        "passes, **threat** for shots on target — FPL counts no passes and no "
        "shots. **Clean sheets** is a team outcome as much as a personal one: a "
        "defender in a well-drilled side collects them whatever he does. "
        "Accurate passes, pass accuracy, long balls, and aerial and ground "
        "duels are absent from the API entirely."
    ),
}


def _profile_section(
    position: str,
    view: pl.DataFrame,
    pool_minutes: int,
    per90: bool,
    sfx: str,
    rank: bool,
) -> None:
    """Draw one position's percentile profile: the matrix, and optionally the
    all-round ranking above it.

    The ranking is orderable by either side as well as by the all-round mean,
    which is what makes it safe for defenders. A defender's all-round score
    rates attacking output as half the job when it is really a bonus, and FPL
    labels every defender "DEF", so the all-round order alone puts overlapping
    full-backs above stay-at-home centre-backs every time. Ranking on
    "Defending" asks the question a centre-back can actually win. The dumbbell
    keeps both sides visible whichever order is chosen, so the trade is never
    hidden.
    """
    profile = queries.POSITION_PROFILES[position]
    sides = queries.profile_sides(position)
    noun = PROFILE_LABEL[position]

    pool = view.filter(pl.col("position") == position)
    if pool.is_empty():
        return

    # Two players can share a web_name, and both charts key rows by the label —
    # an undisambiguated pair would silently collapse into one row of somebody
    # else's numbers. Resolved across the whole position, not per chart, so a
    # player keeps the same label in both.
    labelled = pool.with_columns(
        pl.when(pl.col("web_name").is_duplicated())
        .then(pl.col("web_name") + pl.lit(" · ") + pl.col("team_name"))
        .otherwise(pl.col("web_name"))
        .alias("label")
    )
    cols = {stat: queries.COMPARE_STATS[stat][0] for stat in profile}
    # Percentiles come from the compare board, which ranks within position: 5
    # tackles means nothing next to a centre-back's and everything next to
    # another midfielder's. The pool is every player at the position clearing
    # `queries.peer_pool_minutes` — not the ones left after the filters above
    # it. None of them belong in it. Filtering to one club would rank its
    # players against each other and call the best of them elite; wiring the
    # pool to the minutes slider made a percentile move when a viewer was only
    # decluttering the table, which is the same fault wearing a friendlier hat.
    board = _compare(pool_minutes, per90, METRIC_KEY)
    # Belt as well as braces on the cache key above: whatever the reason a board
    # arrives without a metric's percentile column, say so and draw nothing
    # rather than dying inside a select and taking the whole page with it.
    wanted = [f"{p}_{c}" for c in cols.values() for p in ("p", "v")]
    missing = [c for c in wanted if c not in board.columns]
    if missing:
        st.caption(
            f"The {PROFILE_LABEL[position]} profile is a metric behind the "
            "board it reads from. Clear the cache — press **C**, or *Clear "
            "cache* in the ⋮ menu — and it will rebuild."
        )
        return
    ranked = board.select("fpl_id", *wanted)
    grid = labelled.select("fpl_id", "label", "total_points").join(
        ranked, on="fpl_id", how="left"
    )

    cells_all = pl.DataFrame(
        [
            {
                "player": r["label"],
                "stat": stat,
                "side": side,
                "pct": r[f"p_{cols[stat]}"],
                "value": r[f"v_{cols[stat]}"],
            }
            for r in grid.iter_rows(named=True)
            for stat, side in profile.items()
        ],
        schema={
            "player": pl.Utf8, "stat": pl.Utf8, "side": pl.Utf8,
            "pct": pl.Float64, "value": pl.Float64,
        },
    ).drop_nulls("pct")

    st.markdown(f"**{noun.capitalize()} profile**")
    if cells_all.is_empty():
        st.caption(
            f"No {noun} on screen has played the {pool_minutes} minutes that "
            "put a player in the ranked pool, so there is nothing to draw a "
            "percentile against. Widen the filters to bring in regulars."
        )
        return

    theme = _theme()
    ink = "#52514e" if theme == "light" else "#c3c2b7"
    surface = "#fcfcfb" if theme == "light" else "#1a1a19"
    ramp_domain, ramp = DIVERGING[theme]

    complete = (
        labelled.select("fpl_id", "label")
        .join(
            _pos_scores(position, pool_minutes, per90, METRIC_KEY),
            on="fpl_id",
            how="inner",
        )
        .rename({"label": "player"})
    )

    rank_by = "All-round"
    if rank and not complete.is_empty():
        # Ranking on one side rather than the mean is how a defender board
        # stays honest. FPL calls every defender "DEF", so a single all-round
        # number ranks a centre-back who never leaves his box against an
        # overlapping full-back on a side only one of them is asked to play —
        # and the full-back wins it every time. Picking a side asks the
        # question that actually has an answer. It earns its place for
        # midfielders too: best ball-winner is not best creator.
        rank_by = st.radio(
            "Rank by",
            ["All-round", *sides],
            horizontal=True,
            key=f"rank_by_{position}",
            help=(
                "All-round is the mean of both sides. Pick a side to rank on "
                "that alone — the two are different questions, and for "
                f"{noun}s they can give very different answers."
            ),
        )
    rank_col = "score" if rank_by == "All-round" else rank_by
    complete = complete.sort(rank_col, descending=True, nulls_last=True)

    if rank and not complete.is_empty():
        _profile_ranking(
            complete, sides, rank_col, rank_by, theme, ink, surface, noun
        )

    row_order = "Points"
    if rank and not complete.is_empty():
        row_order = st.radio(
            "Matrix rows",
            ["Points", "Ranking order"],
            horizontal=True,
            key=f"matrix_order_{position}",
            help=(
                "Which players the matrix below draws, and in what order. The "
                "chart above always ranks on the all-round score."
            ),
        )
    if row_order == "Ranking order":
        picked_rows = complete["player"].to_list()[:MATRIX_ROWS]
    else:
        picked_rows = labelled.sort(
            "total_points", descending=True, nulls_last=True
        )["label"].to_list()[:MATRIX_ROWS]

    cells = cells_all.filter(pl.col("player").is_in(picked_rows))
    order = [r for r in picked_rows if r in set(cells["player"])]
    if not order:
        st.caption(f"No {noun} here holds a full profile to draw.")
        return

    cdf = cells.to_pandas()
    base = alt.Chart(cdf).encode(
        x=alt.X(
            "stat:N",
            sort=list(profile),
            title=None,
            axis=alt.Axis(
                orient="top", domain=False, ticks=False, labelColor=ink,
                labelFontSize=11, labelAngle=0, labelPadding=6, labelLimit=180,
            ),
        ),
        y=alt.Y(
            "player:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                domain=False, ticks=False, labelColor=ink,
                labelFontSize=11, labelPadding=8, labelLimit=160,
            ),
        ),
    )
    # The 2px surface stroke is the gap between cells, not a border: drawn in
    # the surface colour it separates without adding a line to read.
    heat = base.mark_rect(
        cornerRadius=3, stroke=surface, strokeWidth=2
    ).encode(
        color=alt.Color(
            "pct:Q",
            title=f"Percentile among {noun}s",
            # Interpolate in RGB explicitly. Vega defaults a continuous colour
            # scale to HCL, and the midpoint here is a near-neutral whose hue is
            # arbitrary — HCL swept from it round to blue through green and
            # clamped out of gamut on the way, so the 60-70 band rendered teal
            # however the stops were chosen.
            scale=alt.Scale(
                domain=ramp_domain, range=ramp, interpolate="rgb", clamp=True
            ),
            legend=alt.Legend(
                orient="bottom", direction="horizontal",
                gradientLength=200, gradientThickness=10,
                labelColor=ink, titleColor=ink,
                labelFontSize=11, titleFontSize=12,
            ),
        ),
        tooltip=[
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("stat:N", title="Metric"),
            alt.Tooltip("value:Q", title=f"Value{sfx}", format=".2f"),
            alt.Tooltip("pct:Q", title="Percentile", format=".0f"),
        ],
    )
    nums = base.mark_text(fontSize=11).encode(
        text=alt.Text("pct:Q", format=".0f"),
        color=(
            alt.value("#0b0b0b")
            if theme == "light"
            else alt.condition(
                f"datum.pct < {DARK_INK_FLIP[0]} || datum.pct > {DARK_INK_FLIP[1]}",
                alt.value("#0b0b0b"),
                alt.value("#ffffff"),
            )
        ),
    )
    st.altair_chart(
        alt.layer(heat, nums)
        .properties(width=alt.Step(104), height=alt.Step(34))
        .facet(
            column=alt.Column(
                "side:N",
                sort=sides,
                title=None,
                header=alt.Header(
                    labelColor=ink, labelFontSize=12,
                    labelFontWeight=600, labelPadding=4,
                ),
            )
        )
        .resolve_scale(x="independent")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelLimit=180),
        width="stretch",
    )
    ranked_on = "points" if row_order == "Points" else rank_by.lower()
    st.caption(
        f"Top {len(order)} {noun}s by {ranked_on}, each metric as a percentile "
        f"against every {noun} with {pool_minutes}+ minutes"
        + (", per 90" if per90 else "")
        + f". Blue is above the median {noun}, red below. "
        + PROFILE_NOTE[position]
    )


def _profile_ranking(
    complete: pl.DataFrame,
    sides: list[str],
    rank_col: str,
    rank_by: str,
    theme: str,
    ink: str,
    surface: str,
    noun: str,
) -> None:
    """The all-round ranking: a dumbbell of the two side scores per player.

    The score is a mean of two numbers, so a bar of it says very little —
    everyone near the top lands between 70 and 90 and the bars come out the same
    length, with the printed number doing all the work. The two sides are what
    actually differ: one man is 98 and 51, the next 71 and 95, and they score
    the same. So plot both ends and let the gap between them be the shape of the
    player; row order and the printed score carry the ranking. The axis stays
    0-100 — cropping it to the occupied range would inflate small gaps.
    """
    board_n = min(RANK_ROWS, complete.height)
    top_all = complete.head(board_n).with_columns(
        pl.max_horizontal(*sides).alias("hi")
    )
    bdf = top_all.to_pandas()
    rank_order = top_all["player"].to_list()
    dumb = top_all.unpivot(
        on=sides, index="player", variable_name="side", value_name="side_pct"
    ).to_pandas()

    ax = dict(
        grid=True, gridOpacity=0.18, gridColor=ink, tickCount=6,
        domain=False, ticks=False, labelColor=ink, titleColor=ink,
        labelFontSize=11, titleFontSize=12, titlePadding=8,
    )
    y_enc = alt.Y(
        "player:N",
        sort=rank_order,
        title=None,
        axis=alt.Axis(
            domain=False, ticks=False, labelColor=ink,
            labelFontSize=11, labelPadding=8, labelLimit=160,
        ),
    )
    # The connector is drawn first and kept faint: it is there to pair the two
    # dots, not to be read itself.
    connector = alt.Chart(bdf).mark_rule(
        strokeWidth=2, opacity=0.35, color=ink
    ).encode(
        y=y_enc,
        x=alt.X(f"{sides[0]}:Q", scale=alt.Scale(domain=[0, 100])),
        x2=alt.X2(f"{sides[1]}:Q"),
    )
    dots = alt.Chart(dumb).mark_circle(
        size=130, opacity=0.95, stroke=surface, strokeWidth=1.5
    ).encode(
        y=y_enc,
        x=alt.X(
            "side_pct:Q",
            title=f"Percentile among {noun}s",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(**ax),
        ),
        color=alt.Color(
            "side:N",
            title=None,
            scale=alt.Scale(domain=sides, range=PALETTE[theme][:2]),
            legend=alt.Legend(
                orient="top", labelColor=ink,
                symbolStrokeWidth=0, labelFontSize=11,
            ),
        ),
        tooltip=[
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("side:N", title="Side"),
            alt.Tooltip("side_pct:Q", title="Percentile", format=".1f"),
        ],
    )
    # The score, printed past whichever dot sits furthest right so it never
    # lands on one.
    rank_nums = alt.Chart(bdf).mark_text(
        align="left", dx=12, fontSize=11, fontWeight=600, color=ink,
    ).encode(
        y=y_enc,
        x=alt.X("hi:Q", scale=alt.Scale(domain=[0, 100])),
        # The number printed is the one the order is built on, so the column
        # always reads as descending. Printing the all-round score while
        # sorting on a side made the list look unsorted.
        text=alt.Text(f"{rank_col}:Q", format=".0f"),
        tooltip=[
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("score:Q", title="All-round", format=".1f"),
            *[alt.Tooltip(f"{s}:Q", title=s, format=".1f") for s in sides],
            alt.Tooltip("flat:Q", title="Flat mean", format=".1f"),
        ],
    )
    heading = (
        f"Most complete {noun}s"
        if rank_by == "All-round"
        else f"Best {noun}s on {rank_by.lower()}"
    )
    st.markdown(f"**{heading}**")
    st.altair_chart(
        (connector + dots + rank_nums)
        # An exact band step rather than a total height: the legend and axis eat
        # into a fixed height and squeezed the rows to about 21px, close enough
        # to run the names together.
        .properties(height=alt.Step(28), padding={"right": 50, "top": 5})
        .configure_view(strokeOpacity=0),
        width="stretch",
    )
    others = " and ".join(s.lower() for s in sides if s != rank_by)
    basis = (
        f"the mean of their two side scores — {sides[0].lower()} and "
        f"{sides[1].lower()} — each side averaged before the two are averaged "
        "together, so a side is not weighted by how many columns it happens to "
        "own. The flat mean is in the tooltip"
        if rank_by == "All-round"
        else f"{rank_by.lower()} alone, ignoring {others} entirely"
    )
    st.caption(
        f"Top {board_n} of {complete.height} {noun}s on {basis}. The number at "
        "the end of each row is what the order is built on, and the gap between "
        f"the two dots is how lopsided the {noun} is. 50 is the median {noun} "
        "on that side, not a midtable finish."
    )


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
            min_minutes = st.slider(
                "Minimum minutes",
                0,
                max(max_min, 1),
                min(90, max_min),
                help=(
                    "Hides players below the bar. It does not change anybody's "
                    "percentile — those are always against the same pool of "
                    f"regulars ({queries.peer_pool_minutes(pb)}+ minutes)."
                ),
            )
        with f4:
            basis = st.radio("Basis", ["Totals", "Per 90"], horizontal=True)

        view = pb.filter(pl.col("minutes") >= min_minutes)
        if pos:
            view = view.filter(pl.col("position").is_in(pos))
        if teams:
            view = view.filter(pl.col("team_name").is_in(teams))

        search = st.text_input("Search player", placeholder="surname…")
        if search:
            view = view.filter(_name_match(search))

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

            # --- position profiles -----------------------------------------
            pool_minutes = queries.peer_pool_minutes(pb)
            _profile_section("MID", view, pool_minutes, per90, sfx, rank=True)
            _profile_section("DEF", view, pool_minutes, per90, sfx, rank=True)

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
        c1, c2 = st.columns([6.5, 2])
        with c2:
            cmp_basis = st.radio(
                "Basis", ["Totals", "Per 90"], horizontal=True, key="cmp_basis"
            )

        # The same pool the Players tab ranks against, so a player reads the
        # same percentile on both. This was a slider, honestly labelled — but
        # an honest label does not stop two tabs disagreeing about a number
        # that has one right answer.
        pool_min = queries.peer_pool_minutes(pool)
        cmp_per90 = cmp_basis == "Per 90"
        board = _compare(pool_min, cmp_per90, METRIC_KEY)
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
            # Two players can share a web_name, and the table below keys its
            # columns by this name: an undisambiguated pair collapsed into one
            # column, which then took two values per metric and threw a
            # ShapeError against the metric list. The club separates them.
            bare = [label.split(" · ")[0] for label in picked]
            short = {
                label: (
                    f"{name} · {rows[label]['team_name']}"
                    if bare.count(name) > 1
                    else name
                )
                for label, name in zip(picked, bare)
            }
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
                    f"Short of the {pool_min} minutes that earn a ranking, so "
                    "unranked: "
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
            "Expected points under BeManager scoring: the Sofascore match "
            "rating converted through the league's band table, with the goal "
            "bonus on top. Every player starts at 6.5 and moves from there, "
            "and the conversion integrates a distribution across the bands "
            "rather than looking up one number — with bands 0.2 wide, a "
            "projection on an edge is worth close to a point either way."
        )

        obs = _observed_ratings()
        rated = int(obs["obs_n"].sum()) if not obs.is_empty() else 0
        if rated == 0:
            st.warning(
                "**Running on priors alone.** Sofascore weights every action "
                "by context, so no fixed tariff built on FPL totals can "
                "reproduce it — and two of its five categories, passing and "
                "dribbling, are absent from this feed entirely. Record ratings "
                "below and each player's own history replaces the guesses for "
                "him. Until then, read the ordering and not the numbers.",
                icon="⚠️",
            )
        else:
            st.success(
                f"{rated} rating(s) recorded across "
                f"{obs.height} player(s) — those players are projected partly "
                f"on their own history rather than on the model's priors.",
                icon="✅",
            )

        with st.expander("What this model can and cannot see"):
            st.caption(
                "Sofascore names sixteen key factors. Against the 109 fields "
                "the FPL API publishes, six have a direct counter, two have "
                "only a proxy, three arrive bundled inside a summed column, "
                "and five have no counter of any kind. Four of those five are "
                "negatives, which is why this model runs optimistic: it sees "
                "most of what lifts a rating and little of what drags one down."
            )
            st.dataframe(
                league.factor_table(), hide_index=True, width="stretch",
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="modelled = a direct counter · proxy = something "
                             "related stands in · bundled = real but summed with "
                             "lesser actions and inseparable · missing = no "
                             "counter anywhere in the feed.",
                    ),
                },
            )
            st.caption(
                "The bundled three — a clearance off the line, a last-man "
                "tackle, a diving save — are the sharpest loss. Each is a "
                "high-value *specific* action that reaches this feed as an "
                "ordinary clearance, tackle or save, so the context Sofascore "
                "weights most heavily is the context the feed flattens."
            )
            st.markdown("**By category**")
            for name, (covered, note) in league.COVERAGE.items():
                st.markdown(f"{'✅' if covered else '❌'} **{name}** — {note}")

        g1, g2, g3 = st.columns([2.2, 2, 2.4])
        with g1:
            gws = st.multiselect(
                "Gameweeks", gw_options, default=gw_options[:1],
                help=(
                    "Pick several to project a run. Totals sum every fixture a "
                    "club has in the range, so a double gameweek counts twice "
                    "and a blank counts as nothing."
                ),
            )
        with g2:
            pos_pick = st.multiselect(
                "Position", ["GKP", "DEF", "MID", "FWD"],
                default=["GKP", "DEF", "MID", "FWD"],
            )
        with g3:
            team_pick = st.multiselect(
                "Team",
                sorted(_players()["team_name"].drop_nulls().unique().to_list()),
                key="proj_teams", help="Empty means every club.",
            )

        g4, g5, g6 = st.columns([2.2, 2, 2.4])
        with g4:
            max_price = st.number_input("Max price £m", 3.5, 20.0, 20.0, step=0.5)
        with g5:
            min_mins = st.number_input(
                "Min minutes played", 0, 3000, 90, step=90,
                help="Season minutes so far. Every rate divides by these, so a "
                     "low bar lets in players whose numbers rest on a cameo.",
            )
        with g6:
            name_search = st.text_input(
                "Search player", placeholder="surname…", key="proj_search"
            )

        rank_by = st.radio(
            "Rank by", LEAGUE_RANKS, horizontal=True, key="proj_rank",
            help=(
                "Ceiling is the chance of a 10+ return, floor the chance of a "
                "negative one. The band table pays down to −4, so a floor is a "
                "real cost and not just missed upside."
            ),
        )

        if not gws:
            st.info("Pick at least one gameweek.")
        else:
            proj = _league_projections(tuple(sorted(gws)))
            if proj.is_empty():
                st.warning(
                    "Nothing to project yet. Club attack and defence ratings "
                    "come from matches already played, so there is nothing to "
                    "build on until the season's first gameweek has finished."
                )
            else:
                named = proj.filter(_name_match(name_search)) if name_search else proj
                view = named.filter(
                    (pl.col("minutes") >= min_mins)
                    & (pl.col("price") <= max_price)
                    & (pl.col("xmins") > 0)
                )
                if pos_pick:
                    view = view.filter(pl.col("position").is_in(pos_pick))
                if team_pick:
                    view = view.filter(pl.col("team_name").is_in(team_pick))

                if view.is_empty():
                    if name_search and not named.is_empty():
                        st.warning(
                            _excluded_note(named, min_mins, max_price, pos_pick, team_pick)
                        )
                    elif name_search:
                        st.warning(f"No player's name contains “{name_search.strip()}”.")
                    else:
                        st.warning("No players match those filters.")
                else:
                    part_cols = list(league.POINT_COMPONENTS)
                    totals = (
                        view.group_by("fpl_id")
                        .agg(
                            pl.col("web_name").first(),
                            pl.col("team_name").first(),
                            pl.col("position").first(),
                            pl.col("price").first(),
                            pl.col("minutes").first(),
                            pl.col("xmins").mean().alias("xmins"),
                            pl.col("selected_by_pct").first(),
                            pl.len().alias("fixtures"),
                            # A rating is per match, so it averages across a
                            # run; the points it produces sum.
                            pl.col("rating_mu").mean().alias("rating_mu"),
                            pl.col("obs_n").first(),
                            pl.col("obs_weight").first(),
                            # Chance of at least one such return in the range,
                            # which is neither the sum nor the mean.
                            (1.0 - (1.0 - pl.col("p_excellent")).product()).alias("p_excellent"),
                            (1.0 - (1.0 - pl.col("p_negative")).product()).alias("p_negative"),
                            pl.concat_str(
                                [pl.col("opponent"), pl.col("venue")], separator=" ("
                            ).add(")").str.join(", ").alias("opponents"),
                            *[pl.col(c).sum() for c in part_cols],
                            pl.col("xp").sum().alias("xp"),
                        )
                        .with_columns(
                            (pl.col("xp") / pl.col("price")).alias("xp_per_m"),
                            pl.when(pl.col("obs_weight") >= 0.6).then(pl.lit("observed"))
                            .when(pl.col("obs_weight") > 0).then(pl.lit("blended"))
                            .otherwise(pl.lit("prior")).alias("basis"),
                        )
                        .sort(LEAGUE_RANK_COL[rank_by],
                              descending=rank_by != "Floor", nulls_last=True)
                    )

                    top_n = min(15, totals.height)
                    top = totals.head(top_n)
                    theme = _theme()
                    ink = "#52514e" if theme == "light" else "#c3c2b7"
                    ring = "#fcfcfb" if theme == "light" else "#1a1a19"

                    stacked = (
                        top.select("web_name", *part_cols)
                        .unpivot(index="web_name", variable_name="part", value_name="pts")
                        .with_columns(
                            pl.col("part")
                            .replace_strict(league.POINT_COMPONENTS, return_dtype=pl.Utf8)
                            .alias("segment")
                        )
                        .filter(pl.col("pts").abs() > 0.005)
                        .with_columns(
                            pl.col("segment").replace_strict(
                                {n: i for i, n in enumerate(LEAGUE_STACK_ORDER)},
                                return_dtype=pl.Int32,
                            ).alias("rank")
                        )
                    )
                    order = top["web_name"].to_list()
                    colors = [LEAGUE_STACK[theme][s] for s in LEAGUE_STACK_ORDER]
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
                                    # Vega thins dense tick labels by default,
                                    # which drops every other player's name.
                                    labelOverlap=False,
                                ),
                            ),
                            x=alt.X("pts:Q", title="Expected points", axis=alt.Axis(**ax)),
                            color=alt.Color(
                                "segment:N", sort=LEAGUE_STACK_ORDER,
                                scale=alt.Scale(domain=LEAGUE_STACK_ORDER, range=colors),
                                legend=alt.Legend(
                                    title=None, orient="top", direction="horizontal",
                                    labelColor=ink, labelFontSize=11, symbolType="square",
                                ),
                            ),
                            order=alt.Order("rank:Q", sort="ascending"),
                            tooltip=[
                                alt.Tooltip("web_name:N", title="Player"),
                                alt.Tooltip("segment:N", title="Component"),
                                alt.Tooltip("pts:Q", title="Points", format=".2f"),
                            ],
                        )
                        .properties(height=max(200, 26 * top_n))
                    )
                    st.altair_chart(bars, width="stretch")
                    st.caption(
                        "Rating is what the performance itself pays, before a "
                        "ball hits the net — for most players it is the whole "
                        "return. A quiet 6.5 is worth 2 points, so the bonus "
                        "only separates the forwards who are actually scoring."
                    )

                    st.dataframe(
                        totals.select(
                            pl.col("web_name").alias("Player"),
                            pl.col("team_name").alias("Club"),
                            pl.col("position").alias("Pos"),
                            pl.col("price").alias("£m"),
                            pl.col("opponents").alias("Fixtures"),
                            pl.col("xmins").round(0).alias("xMins"),
                            pl.col("rating_mu").round(2).alias("Rating"),
                            pl.col("basis").alias("Basis"),
                            pl.col("obs_n").alias("Rated"),
                            (pl.col("p_excellent") * 100).alias("P(10+)"),
                            (pl.col("p_negative") * 100).alias("P(neg)"),
                            pl.col("pts_rating").round(2).alias("Rating pts"),
                            pl.col("pts_goal_bonus").round(2).alias("Goal bonus"),
                            pl.col("xp").round(2).alias("xP"),
                            pl.col("xp_per_m").round(2).alias("xP/£m"),
                        ),
                        hide_index=True, width="stretch",
                        column_config={
                            "Basis": st.column_config.TextColumn(
                                "Basis",
                                help="prior = model guesses only · blended = "
                                     "some recorded ratings · observed = mostly "
                                     "the player's own history.",
                            ),
                            "Rated": st.column_config.NumberColumn(
                                "Rated", help="Ratings recorded for this player.",
                            ),
                            "P(10+)": st.column_config.ProgressColumn(
                                "P(10+)", format="%.0f%%", min_value=0.0, max_value=100.0,
                                help="Chance of an excellent return in the range.",
                            ),
                            "P(neg)": st.column_config.ProgressColumn(
                                "P(neg)", format="%.0f%%", min_value=0.0, max_value=100.0,
                                help="Chance of a negative return. The table pays "
                                     "down to −4, so this is a real cost.",
                            ),
                        },
                    )

        with st.expander("Record what players actually returned", expanded=rated == 0):
            st.caption(
                "The one thing the FPL feed cannot supply, and the only route "
                "from an ordering to a number. Enter the Sofascore rating "
                "wherever you can see it: a recorded rating feeds the "
                "projection directly and starts displacing the model's guesses "
                "for that player straight away, while points alone only help a "
                "later fit."
            )
            last_done = max(1, min(gw_options) - 1)
            rec_gw = st.number_input("Gameweek", 1, 38, last_done, step=1, key="rec_gw")
            blank = pl.DataFrame(
                {"player": ["", "", ""], "rating": [None] * 3, "points": [None] * 3},
                schema={"player": pl.Utf8, "rating": pl.Float64, "points": pl.Int32},
            )
            edited = st.data_editor(
                blank, num_rows="dynamic", hide_index=True, width="stretch",
                key="rec_editor",
                column_config={
                    "player": st.column_config.TextColumn(
                        "Player", help="Surname as it appears on the Players tab."
                    ),
                    "rating": st.column_config.NumberColumn(
                        "Sofascore", min_value=3.0, max_value=10.0, step=0.1,
                        format="%.1f", help="The scale runs 3.0 to 10.0.",
                    ),
                    "points": st.column_config.NumberColumn(
                        "Points", min_value=-4, max_value=40
                    ),
                },
            )
            if st.button("Save returns", key="rec_save"):
                rows, unmatched = _resolve_returns(edited, int(rec_gw))
                if rows:
                    league.record(rows)
                    _observed_ratings.clear()
                    _league_projections.clear()
                    _calibration.clear()
                    st.success(f"Recorded {len(rows)} return(s) for GW{int(rec_gw)}.")
                    st.rerun()
                if unmatched:
                    st.warning(
                        "No player matched: " + ", ".join(unmatched)
                        + ". Names must match the Players tab."
                    )
                if not rows and not unmatched:
                    st.info("Nothing to save — fill in a player and a rating or points.")

            recorded = _calibration()
            if not recorded.is_empty():
                st.caption(f"{recorded.height} return(s) recorded.")
                st.dataframe(recorded, hide_index=True, width="stretch")


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
