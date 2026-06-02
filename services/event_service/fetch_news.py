"""Real A-share news fetching — EastMoney via AKShare + direct API."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from shared.utils.logging import get_logger

logger = get_logger(__name__)


def _stock_news_eastmoney(symbol: str, page: int = 1) -> pd.DataFrame:
    """Fetch stock news from EastMoney search API.

    Direct implementation adapted from FinGPT Inference_datapipe.py.
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    params = {
        "cb": "jQuery3510875346244069884_1668256937995",
        "param": json.dumps({
            "uid": "",
            "keyword": str(symbol),
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": page,
                    "pageSize": 100,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }),
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.eastmoney.com/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    data_text = r.text
    json_str = data_text.strip("jQuery3510875346244069884_1668256937995(").rstrip(")")
    data_json = json.loads(json_str)
    articles = data_json["result"]["cmsArticleWebOld"]
    temp_df = pd.DataFrame(articles)
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    for col in ["新闻标题", "新闻内容"]:
        if col in temp_df.columns:
            temp_df[col] = (
                temp_df[col]
                .astype(str)
                .str.replace(r"<em>", "", regex=True)
                .str.replace(r"</em>", "", regex=True)
                .str.replace(r"　", "", regex=True)
                .str.replace(r"\r\n", " ", regex=True)
            )
    keep_cols = ["新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    result = temp_df[[c for c in keep_cols if c in temp_df.columns]].copy()
    result["关键词"] = symbol
    return result


def fetch_eastmoney_news(symbol: str, max_pages: int = 3) -> pd.DataFrame:
    """Fetch multi-page EastMoney news for a stock symbol."""
    symbol_clean = str(symbol).zfill(6)
    frames = []
    for page in range(1, max_pages + 1):
        try:
            df = _stock_news_eastmoney(symbol_clean, page)
            if df is not None and not df.empty:
                frames.append(df)
        except (KeyError, json.JSONDecodeError, requests.RequestException):
            break
    if not frames:
        return pd.DataFrame()
    news_df = pd.concat(frames, ignore_index=True)
    news_df.drop_duplicates(subset=["新闻标题"], inplace=True)
    return news_df


def fetch_akshare_news(symbol: str, limit: int = 50) -> list[dict]:
    """Fetch news via AKShare stock_news_em (synchronous fallback)."""
    try:
        import akshare as ak

        symbol_6 = str(symbol).zfill(6)
        df = ak.stock_news_em(symbol=symbol_6)
        if df is None or df.empty:
            return []
        news_list = []
        for _, row in df.head(limit).iterrows():
            news_list.append({
                "title": str(row.get("新闻标题", "") or row.get("标题", "")),
                "content": str(row.get("新闻内容", "") or row.get("内容", "")),
                "source": str(row.get("文章来源", "") or "东方财富"),
                "url": str(row.get("新闻链接", "") or row.get("链接", "")),
                "publish_time": str(row.get("发布时间", "") or row.get("时间", "")),
                "symbol": symbol,
            })
        return news_list
    except Exception as exc:
        logger.warning("akshare fallback failed for %s: %s", symbol, exc)
        return []


def fetch_all_news(
    symbols: list[str],
    *,
    max_pages: int = 2,
    limit_per_stock: int = 30,
) -> list[dict]:
    """Fetch news for multiple stocks, merge and deduplicate.

    Returns a list of news dicts with keys:
        title, content, source, url, publish_time, symbols
    """
    all_news: list[dict] = []
    seen_titles: set[str] = set()

    for symbol in symbols:
        try:
            df = fetch_eastmoney_news(symbol, max_pages=max_pages)
            if df is None or df.empty:
                # Fallback to akshare
                akshare_news = fetch_akshare_news(symbol, limit=limit_per_stock)
                for item in akshare_news:
                    key = item["title"][:80]
                    if key not in seen_titles:
                        seen_titles.add(key)
                        item.setdefault("symbols", [symbol])
                        all_news.append(item)
                continue

            for _, row in df.head(limit_per_stock).iterrows():
                title = str(row.get("新闻标题", ""))
                if not title:
                    continue
                key = title[:80]
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                all_news.append({
                    "title": title,
                    "content": str(row.get("新闻内容", "")),
                    "source": str(row.get("文章来源", "") or "东方财富"),
                    "url": str(row.get("新闻链接", "")),
                    "publish_time": str(row.get("发布时间", "")),
                    "symbols": [symbol],
                })
        except Exception as exc:
            logger.warning("fetch_all_news: symbol=%s failed: %s", symbol, exc)
            continue

    logger.info("fetch_all_news: %d stocks → %d unique news items", len(symbols), len(all_news))
    return all_news
