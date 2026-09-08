"""HTTP client. Polling REST — there is no push feed here.

One provider now. The Premier League is the only competition this app covers,
and FPL carries every metric it needs (xG and xA included) under a single
player id, so there is no second feed and no id-mapping seam to maintain.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .config import FPL_BASE, FPL_HEADERS

log = logging.getLogger(__name__)


class FPL:
    """Unofficial but wide open. No key, no auth, no SLA — cache accordingly."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=FPL_BASE, headers=FPL_HEADERS, timeout=20.0
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, **params) -> dict | list:
        """GET with backoff. These endpoints are unsupported — degrade, never hammer."""
        delay = 1.0
        for _ in range(5):
            try:
                r = await self._client.get(path, params=params)
            except httpx.TransportError as exc:
                log.warning("transport error on %s: %s", path, exc)
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 503):
                wait = float(r.headers.get("Retry-After", delay))
                log.warning("throttled on %s, sleeping %.0fs", path, wait)
                await asyncio.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {path} after retries")

    async def bootstrap(self) -> dict:
        """Every player, team and gameweek. Multi-megabyte — poll it slowly."""
        return await self._get("/bootstrap-static/")

    async def live(self, gw: int) -> dict:
        """Per-player stats for a gameweek, including xG, xA and BPS."""
        return await self._get(f"/event/{gw}/live/")

    async def fixtures(self, gw: int) -> list:
        """Kickoff, match minute, score and finished flags for one gameweek."""
        return await self._get("/fixtures/", event=gw)

    async def schedule(self) -> list:
        """Every fixture of the season, played and unplayed, with difficulty.

        Unfiltered — `?future=1` returns only what is left, and the projection
        needs the played ones too to work out how many matches a club has had.
        """
        return await self._get("/fixtures/")

    async def event_status(self) -> dict:
        """Tells you whether bonus points have settled."""
        return await self._get("/event-status/")
