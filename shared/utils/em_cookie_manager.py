"""EastMoney cookie manager — auto-refresh via headless browser.

Cookies live in .em_cookies.json (NOT .env). Auto-refreshed when:
  - File missing or older than COOKIE_TTL (3 hours)
  - API call returns 403 / empty data
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from shared.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_COOKIE_FILE = _PROJECT_ROOT / ".em_cookies.json"
_COOKIE_TTL = 3 * 3600  # 3 hours
_BROWSER_TIMEOUT = 30000  # 30s
_PAGE_STABILIZE = 5  # wait 5s for JS to generate nid18/gviem


class EastMoneyCookieManager:
    """Singleton that loads, caches, and auto-refreshes EastMoney cookies."""

    def __init__(self):
        self._cookies: dict[str, str] = {}
        self._refreshed_at: float = 0.0
        self._lock = asyncio.Lock()
        self._refresh_cooldown = 600  # don't retry refresh within 10 min of failure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_cookie_string(self) -> str:
        """Return cookie header string, refreshing if expired."""
        cookies = await self._ensure_fresh()
        return "; ".join(f"{k}={v}" for k, v in cookies.items())

    async def get_cookie_dict(self) -> dict[str, str]:
        """Return cookie dict for httpx/requests."""
        return await self._ensure_fresh()

    def mark_stale(self) -> None:
        """Force refresh on next get — call when API returns 403."""
        logger.info("[em_cookie] marked stale (will refresh on next call)")
        self._refreshed_at = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_fresh(self) -> dict[str, str]:
        now = time.time()

        # Fast path: in-memory cache still valid
        if self._cookies and (now - self._refreshed_at) < _COOKIE_TTL:
            return self._cookies

        async with self._lock:
            # Double-check after acquiring lock
            if self._cookies and (now - self._refreshed_at) < _COOKIE_TTL:
                return self._cookies

            # Try loading from file first
            loaded = _load_from_file()
            if loaded and (now - loaded.get("_saved_at", 0)) < _COOKIE_TTL:
                self._cookies = {k: v for k, v in loaded.items() if not k.startswith("_")}
                self._refreshed_at = loaded["_saved_at"]
                logger.info("[em_cookie] loaded %d cookies from file (%.0f min old)",
                            len(self._cookies), (now - self._refreshed_at) / 60)
                return self._cookies

            # Need fresh cookies from browser
            try:
                self._cookies = await self._browser_refresh()
                self._refreshed_at = time.time()
                _save_to_file(self._cookies, self._refreshed_at)
                logger.info("[em_cookie] browser refresh OK — %d cookies saved", len(self._cookies))
            except Exception as exc:
                logger.error("[em_cookie] browser refresh failed: %s", exc)
                # Keep stale cookies if we have them, otherwise empty
                if not self._cookies:
                    logger.warning("[em_cookie] no cookies available — EastMoney API will fail")

            return self._cookies

    async def _browser_refresh(self) -> dict[str, str]:
        """Launch headless Chromium, visit EastMoney, extract cookies."""
        from playwright.async_api import async_playwright

        logger.info("[em_cookie] launching headless browser to refresh cookies ...")
        cookies: dict[str, str] = {}

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0"
                ),
                locale="zh-CN",
            )

            try:
                page = await context.new_page()

                # Step 1: Visit EastMoney homepage to get initial cookies
                logger.info("[em_cookie] navigating to eastmoney.com ...")
                await page.goto("https://www.eastmoney.com/", timeout=_BROWSER_TIMEOUT)
                await asyncio.sleep(2)

                # Step 2: Visit quote page to trigger JS fingerprint generation
                await page.goto("https://quote.eastmoney.com/", timeout=_BROWSER_TIMEOUT)
                await asyncio.sleep(_PAGE_STABILIZE)

                # Step 3: Visit data center to get push2 cookies
                await page.goto("https://data.eastmoney.com/zjlx/", timeout=_BROWSER_TIMEOUT)
                await asyncio.sleep(3)

                # Extract all cookies
                browser_cookies = await context.cookies()
                for c in browser_cookies:
                    cookies[c["name"]] = c["value"]

                logger.info("[em_cookie] extracted %d cookies from browser", len(cookies))

            finally:
                await browser.close()

        return cookies


# ------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------

def _load_from_file() -> dict | None:
    if not _COOKIE_FILE.exists():
        return None
    try:
        with open(_COOKIE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_to_file(cookies: dict[str, str], timestamp: float) -> None:
    data = {**cookies, "_saved_at": timestamp}
    try:
        with open(_COOKIE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[em_cookie] failed to save cookie file: %s", exc)


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_cookie_manager: EastMoneyCookieManager | None = None


def get_cookie_manager() -> EastMoneyCookieManager:
    global _cookie_manager
    if _cookie_manager is None:
        _cookie_manager = EastMoneyCookieManager()
    return _cookie_manager
