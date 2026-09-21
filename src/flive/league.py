"""Expected points under BeManager scoring.

`project.py` models FPL: a sum of paid events, where a goal is five points and
the arithmetic is addition. This league works differently. It converts the
Sofascore match rating through a fixed band table and pays a goal bonus on top,
which changes what is worth owning:

  * Involvement beats end product. The rating assesses a performance rather
    than tallying paid events, so a midfielder who creates and tackles all
    afternoon rates well having scored nothing.
  * A bad game costs you. The table runs to -4, and errors, missed big chances
    and lost challenges all drag a rating down. FPL has no floor below zero.
  * Short appearances are cheap rather than wasted. Sofascore rates a player
    from ten minutes, starting from 6.5 like everyone else, so a substitute who
    does nothing wrong banks the baseline.
  * Goals still matter enormously — the bonus is paid on top of a rating the
    goal has already lifted.

So the model projects a *rating*, converts it through the table, and adds the
bonus. Because the table is a step function, the conversion integrates a
distribution across the bands rather than looking up a point estimate: with
bands 0.2 wide, a projection sitting on an edge is worth close to a point
either way.

    xP = P(features) x E[band points | rating ~ N(mu, sigma)]
         + goal bonus x xG + assist bonus x xA

## What this model can and cannot do

Sofascore publishes the shape of its system but not its weights, and states
that its algorithm "assigns a value to each offensive, defensive, and
goalkeeping action based on its context and impact" — a progressive pass is
not a backwards one. A fixed per-action tariff cannot represent that. So the
structural estimate here is a cold start, not a reconstruction, and it is
wrong in a way no amount of tuning fixes.

It is also blind, and `FACTORS` below says exactly how blind. Sofascore names
sixteen key factors; against the 109 fields bootstrap-static publishes, six
have a direct counter, two have only a proxy, three arrive bundled inside a
summed column, and five have no counter of any kind.

Two things follow. Four of the five missing factors are negatives — errors
leading to goals, penalties conceded, unsuccessful dribbles and lost
challenges — so the model sees most of what lifts a rating and little of what
drags one down, and runs optimistic by construction. And the three bundled
ones are a clearance off the line, a last-man tackle and a diving save: all
high-value *specific* actions that reach this feed as ordinary clearances,
tackles and saves. The context Sofascore weights most heavily is precisely the
context this feed flattens away.

One level up, the same gap by category: of shooting, passing, dribbling,
defending and goalkeeping, the feed carries no pass count, no accuracy, no
dribble attempts and no duels at all.

`mu` therefore blends the structural estimate with the player's *own recorded
ratings*, weighted by how many there are (`config.OBSERVED_PRIOR_MATCHES`).
With nothing recorded it is all prior. By half a season it is mostly the
player, and the guesses in `config.py` have quietly stopped mattering. Getting
returns into `league_scores` is the whole path from "ordering you can read" to
"number you can trust".

Source for every [DOC] constant: https://corporate.sofascore.com/about/rating
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from . import project, store
from .config import (
    ASSIST_BONUS,
    FIT_MIN_ROWS,
    GOAL_BONUS,
    MIN_RATED_MINUTES,
    OBSERVED_PRIOR_MATCHES,
    RATING_BANDS,
    RATING_CEILING,
    RATING_CLEAN_SHEET,
    RATING_FLOOR,
    RATING_PER_ASSIST,
    RATING_PER_BIG_CHANCE,
    RATING_PER_CONCEDED,
    RATING_PER_DEF_ACTION,
    RATING_PER_GOAL,
    RATING_PER_OWN_GOAL,
    RATING_PER_PEN_MISS,
    RATING_PER_PEN_SAVE,
    RATING_PER_RED,
    RATING_PER_SAVE,
    RATING_PER_XG_MISS,
    RATING_PER_YELLOW,
    RATING_SIGMA_BASE,
    RATING_SIGMA_MINUTES,
    RATING_START,
    TOP_BAND_POINTS,
)

# Rating at or above which a return is "excellent" on the league's table (10+),
# and below which it goes negative.
EXCELLENT_FROM = 7.95
NEGATIVE_BELOW = 5.95


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """P(Z <= z) for standard normal Z.

    Abramowitz & Stegun 7.1.26, accurate to ~1.5e-7 — far tighter than the
    weights feeding it deserve, and it keeps the dependencies at numpy, which
    polars already brings.
    """
    a = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    p = 0.3275911
    x = z / math.sqrt(2.0)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a[4] * t + a[3]) * t + a[2]) * t + a[1]) * t + a[0]) * t
    return 0.5 * (1.0 + sign * (1.0 - poly * np.exp(-ax * ax)))


def band_points(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Expected league points from a rating distributed N(mu, sigma).

    The table is a step function, so the expectation is the sum over bands of
    (points in band) x (probability of landing in it). Taking the points for
    E[rating] instead is wrong by a band or more wherever the distribution
    straddles an edge, which at this width is most of them.
    """
    edges = np.array([edge for edge, _ in RATING_BANDS], dtype=float)
    points = np.array([pts for _, pts in RATING_BANDS] + [TOP_BAND_POINTS], dtype=float)
    sigma = np.clip(sigma, 1e-6, None)
    cdf = _norm_cdf((edges[None, :] - mu[:, None]) / sigma[:, None])
    lower = np.concatenate([np.zeros((len(mu), 1)), cdf], axis=1)
    upper = np.concatenate([cdf, np.ones((len(mu), 1))], axis=1)
    return ((upper - lower) * points[None, :]).sum(axis=1)


def _tail(mu: np.ndarray, sigma: np.ndarray, threshold: float, above: bool) -> np.ndarray:
    sigma = np.clip(sigma, 1e-6, None)
    below = _norm_cdf((threshold - mu) / sigma)
    return 1.0 - below if above else below


def _rate(column: str) -> pl.Expr:
    """A season total as a per-90 rate, zero below a full match of evidence."""
    return (
        pl.when(pl.col("minutes") >= 90)
        .then(pl.col(column).fill_null(0.0) * 90.0 / pl.col("minutes"))
        .otherwise(0.0)
        .fill_null(0.0)
    )


def observed_ratings() -> pl.DataFrame:
    """Each player's recorded ratings: mean, count and spread.

    This is the half of `expected_points` that is evidence rather than
    assumption. Empty until returns are entered.
    """
    try:
        scores = store.scan("league_scores").collect()
    except Exception:
        return pl.DataFrame(schema={"fpl_id": pl.Int32, "obs_mean": pl.Float64,
                                    "obs_n": pl.UInt32, "obs_sd": pl.Float64})
    if scores.is_empty() or scores["rating"].null_count() == scores.height:
        return pl.DataFrame(schema={"fpl_id": pl.Int32, "obs_mean": pl.Float64,
                                    "obs_n": pl.UInt32, "obs_sd": pl.Float64})
    return (
        scores.filter(pl.col("rating").is_not_null())
        .group_by("fpl_id")
        .agg(
            pl.col("rating").mean().alias("obs_mean"),
            pl.len().alias("obs_n"),
            pl.col("rating").std().alias("obs_sd"),
        )
    )


def expected_points(
    players: pl.DataFrame, schedule: pl.DataFrame, played: pl.DataFrame
) -> pl.DataFrame:
    """One row per player per upcoming fixture, rating and points broken out.

    Shares its shape and its three inputs with `project.expected_points`, so
    the two are swappable and directly comparable.
    """
    ratings = project.team_ratings(players, played)
    if ratings.is_empty():
        return pl.DataFrame()
    fixtures = project.fixture_goals(schedule, ratings)
    if fixtures.is_empty():
        return pl.DataFrame()

    mins = project.minutes_model(players, played)
    df = mins.join(
        fixtures.drop("xg_pm", "xgc_pm", "attack", "defence"), on="team_id", how="inner"
    )
    if df.is_empty():
        return df

    share = (pl.col("xmins") / 90.0).clip(0.0, 1.0)
    keeper = pl.col("position") == "GKP"
    back = pl.col("position").is_in(["GKP", "DEF"])

    df = df.with_columns(
        (pl.col("xg_per_90").fill_null(0.0) * share * pl.col("attack_multiplier")).alias("xg_next"),
        (pl.col("xa_per_90").fill_null(0.0) * share * pl.col("attack_multiplier")).alias("xa_next"),
        _rate("defensive_contribution").alias("def_per_90"),
        # Sofascore rewards *creating a big chance*. FPL publishes no big-chance
        # count, so creativity — Opta's chance-creation index — stands in,
        # scaled to roughly a chance-per-match magnitude.
        (_rate("creativity") / 30.0).alias("chances_per_90"),
        _rate("saves").alias("saves_own_90"),
        _rate("yellow_cards").alias("yellow_90"),
        _rate("red_cards").alias("red_90"),
        _rate("own_goals").alias("og_90"),
        _rate("penalties_missed").alias("penmiss_90"),
        _rate("penalties_saved").alias("pensave_90"),
        # Big chances missed are an explicit negative. xG minus goals is the
        # only handle the feed offers, and it assumes the shortfall persists at
        # the rate it has run at — a strong claim, and the first thing recorded
        # ratings will challenge.
        (
            pl.when(pl.col("minutes") >= 90)
            .then(
                (pl.col("xg").fill_null(0.0) - pl.col("goals_scored").fill_null(0)).clip(0.0, None)
                * 90.0 / pl.col("minutes")
            )
            .otherwise(0.0).fill_null(0.0)
        ).alias("miss_90"),
        ((-pl.col("xg_against")).exp() * pl.col("p60")).alias("p_clean_sheet"),
        # Sofascore rates from ten minutes, so almost anyone who features gets
        # one. This is the share of a match's worth of actions a player is
        # expected to contribute, used to scale the rate-based terms only.
        (pl.col("xmins") / 90.0).clip(0.0, 1.0).alias("action_share"),
    )

    df = df.with_columns(
        (pl.col("xg_next") * RATING_PER_GOAL).alias("r_goals"),
        (pl.col("xa_next") * RATING_PER_ASSIST).alias("r_assists"),
        (pl.col("chances_per_90") * pl.col("action_share")
         * pl.col("attack_multiplier") * RATING_PER_BIG_CHANCE).alias("r_chances"),
        (pl.col("def_per_90") * pl.col("action_share") * RATING_PER_DEF_ACTION).alias("r_defending"),
        (
            pl.when(back)
            .then(
                pl.col("p_clean_sheet") * RATING_CLEAN_SHEET
                + pl.col("xg_against") * pl.col("action_share") * RATING_PER_CONCEDED
            ).otherwise(0.0)
        ).alias("r_keeping"),
        (
            pl.when(keeper)
            .then(
                pl.col("saves_own_90") * pl.col("action_share")
                * pl.col("defence_multiplier") * RATING_PER_SAVE
                # A named factor in its own right, and the one of the sixteen
                # that was available in the feed and going unused.
                + pl.col("pensave_90") * pl.col("action_share") * RATING_PER_PEN_SAVE
            )
            .otherwise(0.0)
        ).alias("r_saves"),
        (pl.col("miss_90") * pl.col("action_share") * RATING_PER_XG_MISS).alias("r_misses"),
        (
            (
                pl.col("yellow_90") * RATING_PER_YELLOW
                + pl.col("red_90") * RATING_PER_RED
                + pl.col("og_90") * RATING_PER_OWN_GOAL
                + pl.col("penmiss_90") * RATING_PER_PEN_MISS
            ) * pl.col("action_share")
        ).alias("r_penalties"),
    )

    parts = ["r_goals", "r_assists", "r_chances", "r_defending",
             "r_keeping", "r_saves", "r_misses", "r_penalties"]
    df = df.with_columns(
        # [DOC] Everyone starts at 6.5 and moves from it.
        pl.lit(RATING_START).alias("r_start"),
        pl.sum_horizontal(parts).alias("r_delta"),
    ).with_columns(
        (pl.col("r_start") + pl.col("r_delta"))
        .clip(RATING_FLOOR, RATING_CEILING).alias("structural_mu")
    )

    # Blend in whatever the player's own recorded ratings say. With none, the
    # weight is zero and this is the structural estimate alone.
    obs = observed_ratings()
    if obs.is_empty():
        df = df.with_columns(
            pl.lit(None, pl.Float64).alias("obs_mean"),
            pl.lit(0, pl.UInt32).alias("obs_n"),
        )
    else:
        df = df.join(obs.select("fpl_id", "obs_mean", "obs_n"), on="fpl_id", how="left")
        df = df.with_columns(pl.col("obs_n").fill_null(0))

    df = df.with_columns(
        (
            pl.col("obs_n").cast(pl.Float64)
            / (pl.col("obs_n").cast(pl.Float64) + OBSERVED_PRIOR_MATCHES)
        ).fill_null(0.0).alias("obs_weight")
    ).with_columns(
        (
            pl.col("obs_weight") * pl.col("obs_mean").fill_null(RATING_START)
            + (1.0 - pl.col("obs_weight")) * pl.col("structural_mu")
        ).clip(RATING_FLOOR, RATING_CEILING).alias("rating_mu"),
        (
            RATING_SIGMA_BASE + RATING_SIGMA_MINUTES * (pl.col("xmins") / 90.0).clip(0.0, 1.0)
        ).alias("rating_sigma"),
    )

    mu = df["rating_mu"].to_numpy()
    sigma = df["rating_sigma"].to_numpy()
    df = df.with_columns(
        pl.Series("pts_rating_raw", band_points(mu, sigma)),
        pl.Series("p_excellent", _tail(mu, sigma, EXCELLENT_FROM, above=True)),
        pl.Series("p_negative", _tail(mu, sigma, NEGATIVE_BELOW, above=False)),
    )

    # [DOC] Ten minutes are needed for a rating at all, so a player who is not
    # expected to reach them has no rating to convert.
    rated = pl.col("xmins") >= MIN_RATED_MINUTES
    return df.with_columns(
        pl.when(rated)
        .then(pl.col("p_play") * pl.col("pts_rating_raw"))
        .otherwise(0.0)
        .alias("pts_rating"),
        (pl.col("xg_next") * GOAL_BONUS).alias("pts_goal_bonus"),
        (pl.col("xa_next") * ASSIST_BONUS).alias("pts_assist_bonus"),
    ).with_columns(
        pl.sum_horizontal("pts_rating", "pts_goal_bonus", "pts_assist_bonus").alias("xp")
    ).sort("xp", descending=True, nulls_last=True)


# Rating components in the order the UI stacks them, mapped to the factors
# Sofascore names. Everything below the baseline is something it calls a
# positive; the last two are the only negatives the FPL feed can see.
RATING_COMPONENTS: dict[str, str] = {
    "r_start": "Baseline 6.5",
    "r_goals": "Goals",
    "r_assists": "Assists",
    "r_chances": "Chances created",
    "r_defending": "Defending",
    "r_keeping": "Clean sheet",
    "r_saves": "Saves",
    "r_misses": "Chances missed",
    "r_penalties": "Cards & errors",
}
POINT_COMPONENTS: dict[str, str] = {
    "pts_rating": "Rating",
    "pts_goal_bonus": "Goal bonus",
    "pts_assist_bonus": "Assist bonus",
}

# Sofascore's sixteen named key factors against what this feed can supply.
# Status is one of:
#   modelled — a direct counter exists and the model uses it
#   proxy    — no counter; something related stands in, with a guessed weight
#   bundled  — the action is real but arrives summed with lesser ones and
#              cannot be separated, so its extra value is lost
#   missing  — no counter of any kind anywhere in the FPL API
#
# The tally is six modelled, two proxied, three bundled and five missing. Four
# of the five missing are negatives, which is the mechanical reason this model
# runs optimistic: it can see most of what lifts a rating and little of what
# drags one down.
FACTORS: list[tuple[str, str, str, str]] = [
    ("+", "Scoring a goal", "modelled", "xG x fixture multiplier"),
    ("+", "Providing an assist", "modelled", "xA x fixture multiplier"),
    ("+", "Creating a big chance", "proxy", "creativity — no big-chance counter"),
    ("+", "Penalty save", "modelled", "penalties_saved"),
    ("+", "Clearance off the line", "bundled", "inside clearances_blocks_interceptions"),
    ("+", "Successful last-man tackle", "bundled", "inside tackles"),
    ("+", "Penalty awarded", "missing", "no counter in the FPL API"),
    ("+", "Successful diving save", "bundled", "inside saves"),
    ("-", "Receiving a red card", "modelled", "red_cards"),
    ("-", "Scoring an own goal", "modelled", "own_goals"),
    ("-", "Committing a penalty", "missing", "no counter in the FPL API"),
    ("-", "Error leading to a goal", "missing", "no counter in the FPL API"),
    ("-", "Big chance missed", "proxy", "xG minus goals — no big-chance counter"),
    ("-", "Penalty miss", "modelled", "penalties_missed"),
    ("-", "Unsuccessful dribbles", "missing", "no counter in the FPL API"),
    ("-", "Lost challenges", "missing", "no counter in the FPL API"),
]

# The same thing one level up: Sofascore's five scoring categories.
COVERAGE: dict[str, tuple[bool, str]] = {
    "Shooting": (True, "xG, goals and threat"),
    "Defending": (True, "tackles, recoveries, clearances, blocks, interceptions"),
    "Goalkeeping": (True, "saves, penalty saves, goals conceded, clean sheets"),
    "Passing": (False, "no pass count or accuracy anywhere in the FPL API"),
    "Dribbling": (False, "no dribble or duel counter anywhere in the FPL API"),
}


def factor_table() -> pl.DataFrame:
    """The sixteen factors and how each is handled, for the UI."""
    return pl.DataFrame(
        [{"": sign, "Factor": name, "Status": status, "Built from": src}
         for sign, name, status, src in FACTORS]
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def record(rows: list[dict]) -> None:
    """Append observed returns. Each row needs gw, fpl_id and points.

    `rating` is optional but is the column that matters: it feeds the blend in
    `expected_points` directly, so a recorded rating starts displacing the
    guesses immediately, while points alone only help a later fit.
    """
    if not rows:
        return
    now = store.now()
    store.write(
        "league_scores",
        store.conform("league_scores", [{"captured_at": now, **r} for r in rows]),
    )


def observed() -> pl.DataFrame:
    """Recorded returns joined to the stat line that produced them."""
    try:
        scores = store.scan("league_scores").collect()
    except Exception:
        return pl.DataFrame()
    if scores.is_empty():
        return scores
    stats = (
        store.scan("player_snapshots").sort("captured_at")
        .group_by(["gw", "fpl_id"]).agg(pl.all().last()).collect()
    )
    if stats.is_empty():
        return pl.DataFrame()
    return scores.join(stats.drop("captured_at", "web_name"), on=["gw", "fpl_id"], how="inner")


FIT_TERMS: dict[str, str] = {
    "goals_scored": "RATING_PER_GOAL",
    "assists": "RATING_PER_ASSIST",
    "creativity": "RATING_PER_BIG_CHANCE",
    "defensive_contribution": "RATING_PER_DEF_ACTION",
    "saves": "RATING_PER_SAVE",
    "clean_sheets": "RATING_CLEAN_SHEET",
    "goals_conceded": "RATING_PER_CONCEDED",
    "yellow_cards": "RATING_PER_YELLOW",
    "red_cards": "RATING_PER_RED",
}


def fit() -> dict[str, float] | None:
    """Least-squares weights for the rating model, or None if too little data.

    Worth being clear about what this can achieve. Sofascore weights actions by
    context, so a linear fit over FPL aggregates will not recover its function
    however many rows it gets — and two of its five categories are missing from
    the inputs entirely. What the fit does give is the best linear predictor
    available from this feed, and an RMSE that says plainly how far that falls
    short. The per-player blend in `expected_points` is the better instrument;
    this is for inspecting which terms carry weight.
    """
    data = observed()
    if data.is_empty() or data.height < FIT_MIN_ROWS:
        return None
    frame = data.filter(pl.col("rating").is_not_null() & (pl.col("minutes") > 0))
    if frame.height < FIT_MIN_ROWS:
        return None

    cols = [c for c in FIT_TERMS if c in frame.columns]
    X = np.column_stack(
        [np.ones(frame.height)] + [frame[c].fill_null(0).cast(pl.Float64).to_numpy() for c in cols]
    )
    y = frame["rating"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out: dict[str, float] = {"_intercept": float(beta[0])}
    for name, value in zip(cols, beta[1:]):
        out[FIT_TERMS[name]] = float(value)
    out["_n"] = float(frame.height)
    out["_rmse"] = float(np.sqrt((resid**2).mean()))
    return out


def calibration() -> pl.DataFrame:
    """Recorded returns, for the UI. Empty until any are entered."""
    data = observed()
    if data.is_empty():
        return pl.DataFrame()
    cols = [c for c in ("gw", "web_name", "fpl_id", "rating", "points", "minutes")
            if c in data.columns]
    return data.select(cols).sort(["gw", "points"], descending=[False, True])
