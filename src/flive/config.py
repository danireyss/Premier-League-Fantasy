"""Configuration. Everything tunable lives here."""

from __future__ import annotations

import os
from pathlib import Path

# --- Storage -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("FLIVE_DATA_DIR", "./data")).resolve()

# --- FPL ---------------------------------------------------------------------
# The only provider. Undocumented and unsupported, but free, keyed to a single
# player id, and — since Opta data landed in it — carrying xG, xA and the
# defensive numbers that used to require a paid feed.
FPL_BASE = "https://fantasy.premierleague.com/api"
# FPL rejects the default httpx/requests user agent intermittently. Set a real one.
FPL_HEADERS = {
    "User-Agent": os.getenv(
        "FLIVE_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    ),
    "Accept": "application/json",
}

# --- Loop cadences (seconds) -------------------------------------------------
LIVE_INTERVAL = 60  # gameweek points, xG and match state while matches are on
SLOW_INTERVAL = 6 * 3600  # prices, ownership, injuries
IDLE_INTERVAL = 300  # how often to re-check for kickoffs when nothing is live

# Buffer ticks in memory and flush periodically. Writing a parquet file every
# tick produces thousands of tiny files a day and makes scans crawl.
FLUSH_EVERY = 60  # seconds

# --- Live stat fields --------------------------------------------------------
# FPL returns the numeric-looking ones as strings; parse.py coerces them.
FLOAT_STATS = {
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goal_involvements": "xgi",
    "expected_goals_conceded": "xgc",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
}

INT_STATS = {
    "minutes": "minutes",
    "total_points": "total_points",
    "bonus": "bonus",
    "bps": "bps",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "saves": "saves",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "tackles": "tackles",
    "recoveries": "recoveries",
    "clearances_blocks_interceptions": "cbi",
    "defensive_contribution": "defensive_contribution",
    "starts": "starts",
}


# --- Projection model --------------------------------------------------------
# FPL's 2025/26 scoring. Points per goal and per clean sheet depend on position;
# everything else is flat.
GOAL_POINTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
SAVES_PER_POINT = 3  # a goalkeeper scores 1 point per 3 saves
CONCEDED_PER_MINUS = 2  # GKP and DEF lose 1 point per 2 goals conceded
DEFCON_POINTS = 2
# Defensive contributions needed in a match to earn those 2 points. Defenders
# count clearances, blocks, interceptions and tackles; everyone else also
# counts ball recoveries, and needs two more of them.
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

# Home advantage, applied to the attacking side's expected goals. The Premier
# League's long-run home scoring edge is roughly ten percent either way. FPL's
# own difficulty rating is already venue-aware, so this is deliberately the
# only place venue is modelled — applying both would count it twice.
HOME_ATTACK = 1.10
AWAY_ATTACK = 0.90

# Club attack and defence ratings are league-relative and derived from a handful
# of matches early on, where one thrashing swings a rating wildly. Shrink each
# toward the league average as though the club had also played PRIOR_MATCHES
# average games. At 3 played and 5 prior, a rating keeps 3/8 of its observed
# distance from average — heavy, and correct, at that sample size.
PRIOR_MATCHES = 5.0

# Minutes model. A named starter almost always reaches the 60 minutes that pay
# the second appearance point, but not quite always — substitutions, injuries
# and red cards take a share. And a player averaging a full BENCH_CERTAIN
# minutes per match is treated as certain to get on the pitch at all.
START_REACHES_60 = 0.88
BENCH_CERTAIN = 20.0

# FPL's availability flags. 'a' is fit; the rest are degrees of doubt, and
# chance_of_playing_next_round overrides these whenever FPL publishes one.
STATUS_AVAILABILITY = {
    "a": 1.0,   # available
    "d": 0.75,  # doubtful
    "i": 0.0,   # injured
    "s": 0.0,   # suspended
    "u": 0.0,   # unavailable
    "n": 0.0,   # not in the squad
}


# --- BeManager scoring -------------------------------------------------------
# The league this app is actually played in does not score events the way FPL
# does. It converts the Sofascore match rating through a fixed band table and
# pays a goal bonus on top.
#
# Constants below are marked [DOC] where Sofascore publishes the value, and
# [PRIOR] where it does not and the number here is a guess. Sofascore's rating
# is a machine-learning model that weights every action by context and impact —
# their own example is a progressive pass against a backwards one — so a fixed
# per-action tariff cannot reproduce it even in principle. Everything marked
# [PRIOR] is therefore a cold-start placeholder, and `league.fit` replaces the
# whole structural estimate with a player's own recorded ratings as they
# accumulate. Read a projection built only on priors as an ordering, not a
# number.
#
# Source: https://corporate.sofascore.com/about/rating

# Rating band -> points, from the league's own table. Written as (exclusive
# upper bound, points). Bounds sit 0.05 below each published edge because
# ratings are quoted to one decimal, so a published "7.0 - 7.1" band covers
# [6.95, 7.15) on a continuous scale.
RATING_BANDS: list[tuple[float, int]] = [
    (4.95, -4),   # 0.0 - 4.9
    (5.35, -3),   # 5.0 - 5.3
    (5.75, -2),   # 5.4 - 5.7
    (5.95, -1),   # 5.8 - 5.9
    (6.15, 0),    # 6.0 - 6.1
    (6.35, 1),    # 6.2 - 6.3
    (6.55, 2),    # 6.4 - 6.5
    (6.75, 3),    # 6.6 - 6.7
    (6.95, 4),    # 6.8 - 6.9
    (7.15, 5),    # 7.0 - 7.1
    (7.35, 6),    # 7.2 - 7.3
    (7.55, 7),    # 7.4 - 7.5
    (7.75, 8),    # 7.6 - 7.7
    (7.95, 9),    # 7.8 - 7.9
    (8.55, 10),   # 8.0 - 8.5
    (9.25, 11),   # 8.6 - 9.2
]
TOP_BAND_POINTS = 12  # 9.3 - 10.0

# [DOC] Every player starts here and moves from it. Note what this implies for
# the band table above: an uneventful game is 6.5, which is 2 points, and the
# bottom two bands need a rating at or below 5.3 — an actively bad shift, not a
# quiet one.
RATING_START = 6.5
# [DOC] The scale's own limits. Nothing reaches 0, so the -4 band is far
# narrower in practice than it looks.
RATING_FLOOR = 3.0
RATING_CEILING = 10.0
# [DOC] "10 minutes of play is needed to generate the first ratings." Below
# this a player has no rating at all rather than a poor one.
MIN_RATED_MINUTES = 10

# [PRIOR] Points paid per goal on top of the rating conversion, backed out of
# four observed Haaland returns against what the band table alone could pay.
# Four matches cannot separate this from the rating lift a goal also produces,
# so it is the least trustworthy number here.
GOAL_BONUS = 4.0
ASSIST_BONUS = 1.5

# [PRIOR] Rating movement per expected event. Sofascore names sixteen key
# factors but publishes no weights, and its real ones are context-dependent
# rather than fixed. `league.FACTORS` maps all sixteen to what this feed can
# supply; the summary is that of 109 element fields in bootstrap-static, six
# factors have a direct counter, two have only a proxy, three are bundled
# inside a summed column that cannot be pulled apart, and five have no counter
# of any kind.
#
# The five with nothing — penalties won, penalties conceded, errors leading to
# goals, unsuccessful dribbles and lost challenges — are four negatives and one
# positive, which is why the model is structurally optimistic. Three of the
# bundled three are high-value *specific* actions (a clearance off the line, a
# last-man tackle, a diving save) that arrive here as ordinary clearances,
# tackles and saves, so exactly the context Sofascore weights heaviest is the
# context this feed flattens.
RATING_PER_GOAL = 0.80
RATING_PER_ASSIST = 0.52
RATING_PER_BIG_CHANCE = 0.22    # creativity stands in; FPL has no big-chance count
RATING_PER_DEF_ACTION = 0.019   # tackles + recoveries + clearances/blocks/int.
RATING_PER_SAVE = 0.055
# A penalty save is a named factor of its own and a far bigger moment than a
# routine stop, so it is weighted separately rather than as one more save.
RATING_PER_PEN_SAVE = 0.85
RATING_CLEAN_SHEET = 0.22
RATING_PER_CONCEDED = -0.105
# Not one of Sofascore's sixteen named factors, but a booking plainly bears on
# a performance and their list is of "key" factors rather than of everything.
# Kept, and flagged here because it is the one term with no documentary basis.
RATING_PER_YELLOW = -0.26
RATING_PER_RED = -1.25
RATING_PER_OWN_GOAL = -0.90
RATING_PER_PEN_MISS = -0.70
# Big chances missed are an explicit negative and xG minus goals is the only
# handle the feed offers. Sized from Sofascore docking roughly a third of a
# point for one, a big chance running about 0.35 xG, and perhaps 40% of a
# striker's shortfall coming from chances big enough to be flagged.
RATING_PER_XG_MISS = -0.33

# No term for the match result. Sofascore does not say whether a team's result
# moves an individual rating, and an earlier version of this model asserted
# that it did with a weight of its own. Absent evidence, it is left out.

# [PRIOR] Spread of a rating around its expectation, used to integrate the
# band table rather than look up a point estimate.
RATING_SIGMA_BASE = 0.30
RATING_SIGMA_MINUTES = 0.27

# How fast a player's own recorded ratings displace the structural estimate
# above. At 4, a player with four recorded ratings is weighted half his own
# history and half the model. This is the mechanism that makes the priors
# temporary rather than permanent.
OBSERVED_PRIOR_MATCHES = 4.0
# Recorded rows needed before `league.fit` will report fitted weights at all.
FIT_MIN_ROWS = 120
