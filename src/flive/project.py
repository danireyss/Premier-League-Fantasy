"""Expected points for a fixture that has not been played.

Everything else in this app reports what happened. This module is the one place
that guesses, so it keeps its assumptions in the open: every constant lives in
config.py, every component of a projection is returned as its own column, and
nothing is folded into a single number the caller cannot take apart.

The shape of the estimate is

    xP  =  appearance + goals + assists + clean sheet + defensive contribution
           + saves + bonus - goals conceded - cards

built from three inputs — a club's attacking and defensive strength, the
fixture it faces, and how much of it the player is likely to be on the pitch
for. FPL publishes an `ep_next` of its own; this deliberately does not look at
it, so the two can be compared rather than one echoing the other.

A caveat worth stating plainly: rates are season-to-date, so in August they
rest on a handful of matches. `team_ratings` shrinks club ratings toward the
league average to stop one thrashing dominating, but nothing can rescue a
player rate built from 90 minutes. Read the projection alongside the minutes.
"""

from __future__ import annotations

import math

import polars as pl

from .config import (
    ASSIST_POINTS,
    AWAY_ATTACK,
    BENCH_CERTAIN,
    CLEAN_SHEET_POINTS,
    CONCEDED_PER_MINUS,
    DEFCON_POINTS,
    DEFCON_THRESHOLD,
    GOAL_POINTS,
    HOME_ATTACK,
    PRIOR_MATCHES,
    SAVES_PER_POINT,
    START_REACHES_60,
    STATUS_AVAILABILITY,
)

# On the pitch at any moment, and so the divisor that turns a squad's summed
# expected-goals-conceded into the club's own. Each player's `xgc` is the xG
# his team conceded while he was playing, so summing the squad counts every
# conceded chance once per player on the pitch for it.
ON_PITCH = 11


def _poisson_at_least(lam: pl.Expr, k: int) -> pl.Expr:
    """P(X >= k) for X ~ Poisson(lam), as an expression.

    Written out as one minus the first k terms rather than reached for from a
    stats library: k is 10 or 12 here, the series is exact at that length, and
    it keeps the whole model inside Polars.
    """
    terms = pl.lit(0.0)
    for i in range(k):
        terms = terms + lam.pow(i) / math.factorial(i)
    return (1.0 - (-lam).exp() * terms).clip(0.0, 1.0)


def _availability() -> pl.Expr:
    """How likely the player is to be fit and in the squad at all.

    `chance_next_round` is FPL's own percentage and wins whenever it is
    published; the status flag is the fallback. The two disagree often — a
    player can sit at status 'a' with a 75% chance after a knock — and the
    explicit number is the better information.
    """
    from_status = pl.col("status").replace_strict(
        STATUS_AVAILABILITY, default=1.0, return_dtype=pl.Float64
    )
    return (
        pl.when(pl.col("chance_next_round").is_not_null())
        .then(pl.col("chance_next_round") / 100.0)
        .otherwise(from_status)
        .fill_null(1.0)
        .clip(0.0, 1.0)
    )


def team_ratings(players: pl.DataFrame, played: pl.DataFrame) -> pl.DataFrame:
    """Attack and defence strength per club, relative to the league average.

    Both sides come from player totals rather than from FPL's own team strength
    fields, which sit at 0 for most of a season:

      * attack — a club's xG is exactly the sum of its players' xG, since every
        chance belongs to whoever took it.
      * defence — each player's `xgc` is what his team conceded while he was on
        the pitch, so the squad's sum counts each chance eleven times, once per
        player out there for it. Divide by eleven and the club's own figure
        falls out. Checked against clubs whose first-choice keeper has played
        every minute, where his `xgc` must equal the club total: it does.

    Ratings are then shrunk toward 1.0 in proportion to matches played, so a
    club three games into a season is mostly the league average and only earns
    its own rating as evidence accumulates.

    `played` is team_id -> matches, which has to come from the fixture list;
    minutes are no use for it, since a club with a sending-off has fewer.
    """
    if players.is_empty() or played.is_empty():
        return pl.DataFrame()

    totals = (
        players.group_by("team_id")
        .agg(
            pl.col("xg").sum().alias("team_xg"),
            (pl.col("xgc").sum() / ON_PITCH).alias("team_xgc"),
        )
        .join(played, on="team_id", how="inner")
        .filter(pl.col("matches") > 0)
    )
    if totals.is_empty():
        return pl.DataFrame()

    totals = totals.with_columns(
        (pl.col("team_xg") / pl.col("matches")).alias("xg_pm"),
        (pl.col("team_xgc") / pl.col("matches")).alias("xgc_pm"),
    )

    # The league average is the baseline both ratings are expressed against and
    # the scale every projected scoreline is built back up from.
    league_xg = totals["xg_pm"].mean() or 1.0
    league_xgc = totals["xgc_pm"].mean() or 1.0

    weight = pl.col("matches") / (pl.col("matches") + PRIOR_MATCHES)
    return totals.with_columns(
        pl.lit(league_xg).alias("league_xg"),
        pl.lit(league_xgc).alias("league_xgc"),
        (1.0 + weight * (pl.col("xg_pm") / league_xg - 1.0)).alias("attack"),
        (1.0 + weight * (pl.col("xgc_pm") / league_xgc - 1.0)).alias("defence"),
    )


def fixture_goals(schedule: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Expected goals for and against, per club per fixture.

    Two rows per fixture, one from each club's point of view, so a player only
    ever has to join on his own team. The scoreline is the league's average
    output scaled by the attacker's strength, the defender's leakiness and the
    venue.

    Venue is applied exactly once, here. FPL's difficulty rating is also
    venue-aware, which is why the projection uses it for display only — putting
    both into the arithmetic would count home advantage twice.
    """
    if schedule.is_empty() or ratings.is_empty():
        return pl.DataFrame()

    rate = ratings.select("team_id", "attack", "defence", "xg_pm", "xgc_pm")
    league_xg = ratings["league_xg"][0]

    sides = []
    for venue, own, opp, own_diff in (
        ("H", "home_team_id", "away_team_id", "home_difficulty"),
        ("A", "away_team_id", "home_team_id", "away_difficulty"),
    ):
        home = venue == "H"
        sides.append(
            schedule.select(
                pl.col("fixture_id"),
                pl.col("gw"),
                pl.col("kickoff_time"),
                pl.col(own).alias("team_id"),
                pl.col(opp).alias("opponent_id"),
                pl.lit(venue).alias("venue"),
                pl.col(own_diff).alias("difficulty"),
                (
                    pl.col("away_name") if home else pl.col("home_name")
                ).alias("opponent"),
            )
        )
    both = pl.concat(sides)

    return (
        both.join(rate, on="team_id", how="inner")
        .join(
            rate.select(
                pl.col("team_id").alias("opponent_id"),
                pl.col("attack").alias("opp_attack"),
                pl.col("defence").alias("opp_defence"),
            ),
            on="opponent_id",
            how="inner",
        )
        .with_columns(
            attack_venue=pl.when(pl.col("venue") == "H")
            .then(HOME_ATTACK)
            .otherwise(AWAY_ATTACK),
        )
        .with_columns(
            # This club's expected goals, and its opponent's — the opponent's
            # venue factor is the mirror of ours.
            (league_xg * pl.col("attack") * pl.col("opp_defence") * pl.col("attack_venue"))
            .alias("xg_for"),
            (
                league_xg
                * pl.col("opp_attack")
                * pl.col("defence")
                * (HOME_ATTACK + AWAY_ATTACK - pl.col("attack_venue"))
            ).alias("xg_against"),
        )
        .with_columns(
            # How much easier or harder than this club's own season average the
            # fixture is. Player rates are season-to-date, so this is the factor
            # that carries the opponent and the venue into them.
            pl.when(pl.col("xg_pm") > 0)
            .then(pl.col("xg_for") / pl.col("xg_pm"))
            .otherwise(1.0)
            .alias("attack_multiplier"),
            pl.when(pl.col("xgc_pm") > 0)
            .then(pl.col("xg_against") / pl.col("xgc_pm"))
            .otherwise(1.0)
            .alias("defence_multiplier"),
        )
    )


def minutes_model(players: pl.DataFrame, played: pl.DataFrame) -> pl.DataFrame:
    """Expected minutes, and the two probabilities appearance points turn on.

    FPL pays one point for appearing and a second for reaching 60 minutes, so
    the two have to be modelled separately — a regular substitute is worth one
    of them and never the other.

    Everything is per *club match*, not per appearance: a player who has missed
    half his side's games should read as half-time-ish, and dividing by his own
    appearances would hide exactly that.
    """
    joined = players.join(played, on="team_id", how="left").with_columns(
        pl.col("matches").fill_null(0)
    )
    per_match = pl.when(pl.col("matches") > 0).then(
        pl.col("minutes") / pl.col("matches")
    ).otherwise(0.0)
    start_rate = (
        pl.when(pl.col("matches") > 0)
        .then(pl.col("starts") / pl.col("matches"))
        .otherwise(0.0)
        .clip(0.0, 1.0)
    )

    return joined.with_columns(
        _availability().alias("availability"),
        per_match.alias("mins_per_match"),
        start_rate.alias("start_rate"),
    ).with_columns(
        (pl.col("availability") * pl.col("mins_per_match").clip(0.0, 90.0)).alias("xmins"),
        (pl.col("availability") * pl.col("start_rate") * START_REACHES_60).alias("p60"),
    ).with_columns(
        # Getting on the pitch at all: a starter's chance, or a bench player's,
        # whichever is larger. Averaging a full BENCH_CERTAIN minutes a match
        # counts as certain to feature.
        pl.max_horizontal(
            pl.col("p60"),
            pl.col("availability") * (pl.col("mins_per_match") / BENCH_CERTAIN).clip(0.0, 1.0),
        ).alias("p_play")
    )


def _per_90(column: str) -> pl.Expr:
    """A season total as a rate. Null below a full match, where it is noise."""
    return (
        pl.when(pl.col("minutes") >= 90)
        .then(pl.col(column) * 90.0 / pl.col("minutes"))
        .otherwise(0.0)
        .fill_null(0.0)
    )


def expected_points(
    players: pl.DataFrame, schedule: pl.DataFrame, played: pl.DataFrame
) -> pl.DataFrame:
    """One row per player per upcoming fixture, with xP broken into its parts.

    Per fixture rather than per gameweek on purpose: a double gameweek gives a
    player two rows that sum to his gameweek total, and a blank gives him none,
    both without a special case anywhere.
    """
    ratings = team_ratings(players, played)
    if ratings.is_empty():
        return pl.DataFrame()

    fixtures = fixture_goals(schedule, ratings)
    if fixtures.is_empty():
        return pl.DataFrame()

    mins = minutes_model(players, played)
    df = mins.join(
        fixtures.drop("xg_pm", "xgc_pm", "attack", "defence"), on="team_id", how="inner"
    )
    if df.is_empty():
        return df

    share = pl.col("xmins") / 90.0
    goal_pts = pl.col("position").replace_strict(
        GOAL_POINTS, default=4, return_dtype=pl.Float64
    )
    cs_pts = pl.col("position").replace_strict(
        CLEAN_SHEET_POINTS, default=0, return_dtype=pl.Float64
    )

    df = df.with_columns(
        # Attacking output is the player's own rate, moved by how this fixture
        # compares with the average one his club has already played.
        (pl.col("xg_per_90").fill_null(0.0) * share * pl.col("attack_multiplier")).alias("xg_next"),
        (pl.col("xa_per_90").fill_null(0.0) * share * pl.col("attack_multiplier")).alias("xa_next"),
        # A clean sheet needs the opponent held to nothing *and* 60 minutes on
        # the pitch to be paid for it.
        ((-pl.col("xg_against")).exp() * pl.col("p60")).alias("p_clean_sheet"),
        _per_90("bonus").alias("bonus_per_90"),
        _per_90("yellow_cards").alias("yellow_per_90"),
        _per_90("red_cards").alias("red_per_90"),
    )

    defcon_lambda = pl.col("defensive_contribution_per_90").fill_null(0.0) * share
    defcon_hit = (
        pl.when(pl.col("position") == "DEF")
        .then(_poisson_at_least(defcon_lambda, DEFCON_THRESHOLD["DEF"]))
        .when(pl.col("position").is_in(["MID", "FWD"]))
        .then(_poisson_at_least(defcon_lambda, DEFCON_THRESHOLD["MID"]))
        .otherwise(0.0)  # goalkeepers are not eligible
    )
    keeper = pl.col("position") == "GKP"
    back = pl.col("position").is_in(["GKP", "DEF"])

    df = df.with_columns(
        (pl.col("p_play") + pl.col("p60")).alias("pts_appearance"),
        (pl.col("xg_next") * goal_pts).alias("pts_goals"),
        (pl.col("xa_next") * ASSIST_POINTS).alias("pts_assists"),
        (pl.col("p_clean_sheet") * cs_pts).alias("pts_clean_sheet"),
        (defcon_hit * DEFCON_POINTS).alias("pts_defcon"),
        # Saves scale with what the opponent is expected to create, the same
        # multiplier that makes a clean sheet less likely.
        pl.when(keeper)
        .then(
            pl.col("saves_per_90").fill_null(0.0)
            * share
            * pl.col("defence_multiplier")
            / SAVES_PER_POINT
        )
        .otherwise(0.0)
        .alias("pts_saves"),
        (pl.col("bonus_per_90") * share).alias("pts_bonus"),
        # FPL docks a point per two conceded, so the expected cost is half the
        # goals a keeper or defender is on the pitch for. Linear, where the real
        # rule steps every second goal — close enough at these totals, and it
        # errs the same way for everyone.
        pl.when(back)
        .then(-pl.col("xg_against") * share / CONCEDED_PER_MINUS)
        .otherwise(0.0)
        .alias("pts_conceded"),
        (-(pl.col("yellow_per_90") + 3.0 * pl.col("red_per_90")) * share).alias("pts_cards"),
    )

    parts = [
        "pts_appearance", "pts_goals", "pts_assists", "pts_clean_sheet",
        "pts_defcon", "pts_saves", "pts_bonus", "pts_conceded", "pts_cards",
    ]
    return df.with_columns(
        pl.sum_horizontal(parts).alias("xp")
    ).sort("xp", descending=True, nulls_last=True)


# The components in the order they are stacked in the UI: what a player earns
# for turning up, then what he earns for what he does, then what he loses.
COMPONENTS: dict[str, str] = {
    "pts_appearance": "Appearance",
    "pts_goals": "Goals",
    "pts_assists": "Assists",
    "pts_clean_sheet": "Clean sheet",
    "pts_defcon": "Defensive",
    "pts_saves": "Saves",
    "pts_bonus": "Bonus",
    "pts_conceded": "Conceded",
    "pts_cards": "Cards",
}
