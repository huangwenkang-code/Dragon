"""Minimal stock utilities stub."""

from enum import Enum


class StockMarket(Enum):
    A_SHARE = "a_share"
    HK = "hk"
    US = "us"


class StockUtils:
    """Minimal stub — provides stock code normalization."""

    @staticmethod
    def normalize_code(code: str, market: str = "a_share") -> str:
        code = str(code).strip()
        if market == "a_share":
            return code.zfill(6)
        return code

    @staticmethod
    def detect_market(code: str) -> str:
        code = str(code).strip()
        if len(code) == 6 and code.isdigit():
            return "a_share"
        if len(code) == 5 and code.isdigit():
            return "hk"
        return "us"

    @staticmethod
    def get_market_from_stock_code(code: str) -> StockMarket:
        market = StockUtils.detect_market(code)
        return {
            "a_share": StockMarket.A_SHARE,
            "hk": StockMarket.HK,
            "us": StockMarket.US,
        }.get(market, StockMarket.A_SHARE)
