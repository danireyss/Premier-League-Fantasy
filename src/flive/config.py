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
