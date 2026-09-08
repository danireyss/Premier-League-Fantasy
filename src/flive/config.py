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
