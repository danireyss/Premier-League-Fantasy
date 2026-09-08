"""Turn FPL JSON into flat rows.

Everything here is defensive. FPL returns its numeric-looking stats as strings,
omits keys between seasons, and leaves scores null until a match kicks off.
Assume every key can be absent or the wrong type.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import FLOAT_STATS, INT_STATS

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _f(v: Any) -> float | None:
    """FPL sends xG, ICT and friends as strings: "0.61", "", sometimes null."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _ts(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def current_gameweek(bootstrap: dict) -> int | None:
    for ev in bootstrap.get("events", []):
        if ev.get("is_current"):
            return ev["id"]
    for ev in bootstrap.get("events", []):
        if ev.get("is_next"):
            return ev["id"]
    return None


def bonus_is_final(status: dict) -> bool:
    entries = status.get("status") or []
    return bool(entries) and all(e.get("bonus_added") for e in entries)


def team_names(bootstrap: dict) -> dict[int, str]:
    return {t["id"]: t.get("name") for t in bootstrap.get("teams", [])}


def player_meta(bootstrap: dict) -> dict[int, dict]:
    """fpl_id -> team, position, name. Needed to attribute live rows to a side."""
    return {
        el["id"]: {
            "team_id": el.get("team"),
            "position": POSITIONS.get(el.get("element_type")),
            "web_name": el.get("web_name"),
        }
        for el in bootstrap.get("elements", [])
        if el.get("id") is not None
    }


def fixture_state(fx: dict) -> str:
    """FPL has no state enum, only flags. Derive one.

    PROV is the window where the match is over but bonus has not settled —
    points shown during it will still move.
    """
    if not fx.get("started"):
        return "UPCOMING"
    if fx.get("finished"):
        return "FT"
    if fx.get("finished_provisional"):
        return "PROV"
    return "LIVE"


def fixture_rows(
    fixtures: list[dict],
    team_xg: dict[tuple[int, int], float],
    names: dict[int, str],
    captured_at: datetime,
) -> list[dict]:
    """One row per fixture, per tick.

    team_xg is keyed (fixture_id, team_id) and comes from summing the live
    per-player xG of each side — FPL publishes no team-level xG of its own.
    """
    out = []
    for fx in fixtures:
        fid, h, a = fx.get("id"), fx.get("team_h"), fx.get("team_a")
        out.append(
            {
                "captured_at": captured_at,
                "gw": fx.get("event"),
                "fixture_id": fid,
                "kickoff_time": _ts(fx.get("kickoff_time")),
                "state": fixture_state(fx),
                "minute": _i(fx.get("minutes")),
                "home_team_id": h,
                "away_team_id": a,
                "home_name": names.get(h),
                "away_name": names.get(a),
                "home_score": _i(fx.get("team_h_score")),
                "away_score": _i(fx.get("team_a_score")),
                "home_xg": team_xg.get((fid, h)),
                "away_xg": team_xg.get((fid, a)),
            }
        )
    return out


def _fixture_for(el: dict, meta: dict, team_fixture: dict[int, int]) -> int | None:
    """Which match this player's numbers belong to.

    `explain` is authoritative and carries one entry per fixture played, so its
    last entry is the most recent. A player yet to touch the pitch has none, so
    fall back to whichever fixture their club is in this gameweek.
    """
    explain = el.get("explain") or []
    if explain and explain[-1].get("fixture"):
        return explain[-1]["fixture"]
    return team_fixture.get(meta.get("team_id"))


def player_rows(
    live: dict,
    gw: int,
    bonus_final: bool,
    meta_by_id: dict[int, dict],
    team_fixture: dict[int, int],
    captured_at: datetime,
) -> list[dict]:
    """One row per player, per tick.

    Provisional throughout: bps moves all match and bonus only settles after it,
    so bonus_final rides along on every row and the UI warns while it is false.
    """
    out = []
    for el in live.get("elements", []):
        fpl_id = el.get("id")
        stats = el.get("stats") or {}
        meta = meta_by_id.get(fpl_id, {})
        row = {
            "captured_at": captured_at,
            "gw": gw,
            "fpl_id": fpl_id,
            "fixture_id": _fixture_for(el, meta, team_fixture),
            "team_id": meta.get("team_id"),
            "web_name": meta.get("web_name"),
            "position": meta.get("position"),
            "bonus_final": bonus_final,
        }
        for src, dst in INT_STATS.items():
            row[dst] = _i(stats.get(src))
        for src, dst in FLOAT_STATS.items():
            row[dst] = _f(stats.get(src))
        out.append(row)
    return out


def team_xg_from_players(
    live: dict, meta_by_id: dict[int, dict], team_fixture: dict[int, int]
) -> dict[tuple[int, int], float]:
    """Sum live per-player xG into a per-side total, keyed (fixture_id, team_id)."""
    totals: dict[tuple[int, int], float] = {}
    for el in live.get("elements", []):
        meta = meta_by_id.get(el.get("id"), {})
        team_id = meta.get("team_id")
        fixture_id = _fixture_for(el, meta, team_fixture)
        xg = _f((el.get("stats") or {}).get("expected_goals"))
        if team_id is None or fixture_id is None or xg is None:
            continue
        totals[(fixture_id, team_id)] = totals.get((fixture_id, team_id), 0.0) + xg
    return {k: round(v, 3) for k, v in totals.items()}


def team_fixture_map(fixtures: list[dict]) -> dict[int, int]:
    """team_id -> the fixture it is currently in, preferring one in play.

    Double gameweeks put a club in two fixtures; an in-play one is the useful
    answer, and the latest kickoff is the best guess otherwise.
    """
    out: dict[int, int] = {}
    ordered = sorted(
        fixtures, key=lambda f: (bool(f.get("started")) and not f.get("finished"), f.get("kickoff_time") or "")
    )
    for fx in ordered:
        for side in ("team_h", "team_a"):
            if fx.get(side) is not None and fx.get("id") is not None:
                out[fx[side]] = fx["id"]
    return out


def fpl_player_rows(bootstrap: dict, captured_at: datetime) -> list[dict]:
    names = team_names(bootstrap)
    return [
        {
            "captured_at": captured_at,
            "fpl_id": el.get("id"),
            "web_name": el.get("web_name"),
            "full_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
            "team_id": el.get("team"),
            "team_name": names.get(el.get("team")),
            "position": POSITIONS.get(el.get("element_type")),
            "now_cost": el.get("now_cost"),
            "selected_by_pct": _f(el.get("selected_by_percent")),
            "form": _f(el.get("form")),
            "total_points": el.get("total_points"),
            "status": el.get("status"),
            "chance_next_round": el.get("chance_of_playing_next_round"),
            "minutes": _i(el.get("minutes")),
            "starts": _i(el.get("starts")),
            "goals_scored": _i(el.get("goals_scored")),
            "assists": _i(el.get("assists")),
            "bonus": _i(el.get("bonus")),
            "bps": _i(el.get("bps")),
            "xg": _f(el.get("expected_goals")),
            "xa": _f(el.get("expected_assists")),
            "xgi": _f(el.get("expected_goal_involvements")),
            "xgc": _f(el.get("expected_goals_conceded")),
            "influence": _f(el.get("influence")),
            "creativity": _f(el.get("creativity")),
            "threat": _f(el.get("threat")),
            "ict_index": _f(el.get("ict_index")),
            "xg_per_90": _f(el.get("expected_goals_per_90")),
            "xa_per_90": _f(el.get("expected_assists_per_90")),
            "xgi_per_90": _f(el.get("expected_goal_involvements_per_90")),
            "tackles": _i(el.get("tackles")),
            "recoveries": _i(el.get("recoveries")),
            "cbi": _i(el.get("clearances_blocks_interceptions")),
            "defensive_contribution": _i(el.get("defensive_contribution")),
            "points_per_game": _f(el.get("points_per_game")),
            "value_season": _f(el.get("value_season")),
            "ep_next": _f(el.get("ep_next")),
        }
        for el in bootstrap.get("elements", [])
    ]
