"""
Alpha Factor Engineering — standalone factor builder adapted from Microsoft Qlib.

The original Alpha360 and Alpha158 are feature-generation pipelines defined in
qlib.contrib.data.loader.  This module extracts the factor *definitions* and
re-implements them against plain pandas DataFrames so that dragon-engine has no
dependency on the qlib runtime.

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper functions that mirror Qlib expression semantics on pandas
# ---------------------------------------------------------------------------

def _ref(series: pd.Series, lag: int) -> pd.Series:
    """Shift *series* backward by *lag* periods (``Ref`` in Qlib expression)."""
    if lag == 0:
        return series.copy()
    return series.groupby(level=1).shift(lag)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling simple moving average (``Mean`` in Qlib)."""
    return series.groupby(level=1).transform(lambda g: g.rolling(window, min_periods=window).mean())


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation (``Std`` in Qlib)."""
    return series.groupby(level=1).transform(lambda g: g.rolling(window, min_periods=window).std())


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum (``Max`` in Qlib)."""
    return series.groupby(level=1).transform(lambda g: g.rolling(window, min_periods=window).max())


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum (``Min`` in Qlib)."""
    return series.groupby(level=1).transform(lambda g: g.rolling(window, min_periods=window).min())


def _rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile (``Quantile`` in Qlib)."""
    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).quantile(q)
    )


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank within each stock (``Rank`` in Qlib)."""
    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).rank(pct=True)
    )


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Linear-regression slope over rolling window (``Slope`` in Qlib)."""

    def _ols_slope(y: np.ndarray) -> float:
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        return num / den if den != 0 else 0.0

    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).apply(_ols_slope, raw=True)
    )


def _rolling_rsquare(series: pd.Series, window: int) -> pd.Series:
    """Linear-regression R-squared over rolling window (``Rsquare`` in Qlib)."""

    def _ols_r2(y: np.ndarray) -> float:
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        ss_yy = ((y - y_mean) ** 2).sum()
        if ss_xx == 0 or ss_yy == 0:
            return 0.0
        slope = ss_xy / ss_xx
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = ((y - y_pred) ** 2).sum()
        return 1.0 - ss_res / ss_yy

    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).apply(_ols_r2, raw=True)
    )


def _rolling_resi(series: pd.Series, window: int) -> pd.Series:
    """Linear-regression residual (last point) over rolling window (``Resi`` in Qlib)."""

    def _last_residual(y: np.ndarray) -> float:
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx == 0:
            return 0.0
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred = intercept + slope * x[-1]
        return y[-1] - y_pred

    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).apply(_last_residual, raw=True)
    )


def _rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
    """Days since last maximum within window (``IdxMax`` in Qlib)."""
    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).apply(
            lambda a: (len(a) - 1 - np.argmax(a)), raw=True
        )
    )


def _rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
    """Days since last minimum within window (``IdxMin`` in Qlib)."""
    return series.groupby(level=1).transform(
        lambda g: g.rolling(window, min_periods=window).apply(
            lambda a: (len(a) - 1 - np.argmin(a)), raw=True
        )
    )


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between *a* and *b* (``Corr`` in Qlib)."""
    combined = pd.DataFrame({"a": a, "b": b}, index=a.index)
    return combined.groupby(level=1).transform(
        lambda g: g["a"].rolling(window, min_periods=window).corr(g["b"])
    )


# ---------------------------------------------------------------------------
# AlphaFactorBuilder  --  the main entry point
# ---------------------------------------------------------------------------

class AlphaFactorBuilder:
    """Build Alpha-style factor columns from raw OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (datetime, instrument) DataFrame with columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``vwap``.
        All columns must be present in lower-case.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._validate_columns()

    # -- column helpers -------------------------------------------------------
    @property
    def _open(self) -> pd.Series:
        return self._df["open"]

    @property
    def _high(self) -> pd.Series:
        return self._df["high"]

    @property
    def _low(self) -> pd.Series:
        return self._df["low"]

    @property
    def _close(self) -> pd.Series:
        return self._df["close"]

    @property
    def _volume(self) -> pd.Series:
        return self._df["volume"]

    @property
    def _vwap(self) -> pd.Series:
        return self._df.get("vwap", self._close)

    def _validate_columns(self) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(self._df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Alpha360  (360 raw price/volume factors, 60-day lookback)
    # ------------------------------------------------------------------

    def build_alpha360(self) -> pd.DataFrame:
        """Generate Alpha360 factors.

        Returns a DataFrame with columns CLOSE0-59, OPEN0-59, HIGH0-59,
        LOW0-59, VWAP0-59, VOLUME0-59 -- 6 x 60 = 360 features.
        All price columns are normalised by the *latest* close;
        volume columns are normalised by the *latest* volume.
        """
        fields: Dict[str, pd.Series] = {}
        close = self._close
        volume = self._volume

        for name, series in [
            ("CLOSE", close),
            ("OPEN", self._open),
            ("HIGH", self._high),
            ("LOW", self._low),
            ("VWAP", self._vwap),
        ]:
            for lag in range(59, -1, -1):
                col_name = f"{name}{59 - lag}"
                if lag == 0:
                    fields[col_name] = series / close
                else:
                    fields[col_name] = _ref(series, lag) / close

        # Volume features
        for lag in range(59, -1, -1):
            col_name = f"VOLUME{59 - lag}"
            if lag == 0:
                fields[col_name] = volume / (volume + 1e-12)
            else:
                fields[col_name] = _ref(volume, lag) / (volume + 1e-12)

        return pd.DataFrame(fields, index=self._df.index)

    # ------------------------------------------------------------------
    # Alpha158  (up to 158 factors: kbar + price + volume + rolling)
    # ------------------------------------------------------------------

    def build_alpha158(
        self,
        *,
        kbar: bool = True,
        price_windows: Optional[List[int]] = None,
        volume_windows: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_include: Optional[List[str]] = None,
        rolling_exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate Alpha158 factors.

        Parameters
        ----------
        kbar : bool
            Include 9 hard-coded candlestick factors (KMID, KLEN, ...).
        price_windows : list[int] | None
            Lookback lags for price features.  Default ``[0,1,2,3,4]``.
        volume_windows : list[int] | None
            Lookback lags for volume features.  Default ``[0,1,2,3,4]``.
        rolling_windows : list[int] | None
            Rolling window sizes for technical indicators.  Default ``[5,10,20,30,60]``.
        rolling_include : list[str] | None
            Subset of rolling operators to include (default: all).
            Available: ROC, MA, STD, BETA, RSQR, RESI, MAX, LOW, QTLU, QTLD,
                       RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD,
                       SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD.
        rolling_exclude : list[str] | None
            Operators to exclude (applied after *rolling_include*).
        """
        if price_windows is None:
            price_windows = [0, 1, 2, 3, 4]
        if volume_windows is None:
            volume_windows = [0, 1, 2, 3, 4]
        if rolling_windows is None:
            rolling_windows = [5, 10, 20, 30, 60]
        if rolling_exclude is None:
            rolling_exclude = []

        def _use(op: str) -> bool:
            if op in rolling_exclude:
                return False
            if rolling_include is not None:
                return op in rolling_include
            return True

        fields: Dict[str, pd.Series] = {}
        close = self._close
        high = self._high
        low = self._low
        vol = self._volume
        open_ = self._open

        # -- kbar factors -------------------------------------------------
        if kbar:
            kbar_fields = {
                "KMID": (close - open_) / open_,
                "KLEN": (high - low) / open_,
                "KMID2": (close - open_) / (high - low + 1e-12),
                "KUP": (high - np.maximum(open_, close)) / open_,
                "KUP2": (high - np.maximum(open_, close)) / (high - low + 1e-12),
                "KLOW": (np.minimum(open_, close) - low) / open_,
                "KLOW2": (np.minimum(open_, close) - low) / (high - low + 1e-12),
                "KSFT": (2 * close - high - low) / open_,
                "KSFT2": (2 * close - high - low) / (high - low + 1e-12),
            }
            fields.update(kbar_fields)

        # -- price features ------------------------------------------------
        for field in ["OPEN", "HIGH", "LOW", "VWAP"]:
            raw = self._df.get(field.lower(), close)
            for d in price_windows:
                col = f"{field}{d}"
                if d == 0:
                    fields[col] = raw / close
                else:
                    fields[col] = _ref(raw, d) / close

        # -- volume features -----------------------------------------------
        for d in volume_windows:
            col = f"VOLUME{d}"
            if d == 0:
                fields[col] = vol / (vol + 1e-12)
            else:
                fields[col] = _ref(vol, d) / (vol + 1e-12)

        # -- rolling features ----------------------------------------------
        wlist = rolling_windows

        if _use("ROC"):
            for w in wlist:
                fields[f"ROC{w}"] = _ref(close, w) / close

        if _use("MA"):
            for w in wlist:
                fields[f"MA{w}"] = _rolling_mean(close, w) / close

        if _use("STD"):
            for w in wlist:
                fields[f"STD{w}"] = _rolling_std(close, w) / close

        if _use("BETA"):
            for w in wlist:
                fields[f"BETA{w}"] = _rolling_slope(close, w) / close

        if _use("RSQR"):
            for w in wlist:
                fields[f"RSQR{w}"] = _rolling_rsquare(close, w)

        if _use("RESI"):
            for w in wlist:
                fields[f"RESI{w}"] = _rolling_resi(close, w) / close

        if _use("MAX"):
            for w in wlist:
                fields[f"MAX{w}"] = _rolling_max(high, w) / close

        if _use("LOW"):  # note: Qlib calls this MIN in the name
            for w in wlist:
                fields[f"MIN{w}"] = _rolling_min(low, w) / close

        if _use("QTLU"):
            for w in wlist:
                fields[f"QTLU{w}"] = _rolling_quantile(close, w, 0.8) / close

        if _use("QTLD"):
            for w in wlist:
                fields[f"QTLD{w}"] = _rolling_quantile(close, w, 0.2) / close

        if _use("RANK"):
            for w in wlist:
                fields[f"RANK{w}"] = _rolling_rank(close, w)

        if _use("RSV"):
            for w in wlist:
                fields[f"RSV{w}"] = (close - _rolling_min(low, w)) / (
                    _rolling_max(high, w) - _rolling_min(low, w) + 1e-12
                )

        if _use("IMAX"):
            for w in wlist:
                fields[f"IMAX{w}"] = _rolling_idxmax(high, w) / w

        if _use("IMIN"):
            for w in wlist:
                fields[f"IMIN{w}"] = _rolling_idxmin(low, w) / w

        if _use("IMXD"):
            for w in wlist:
                fields[f"IMXD{w}"] = (_rolling_idxmax(high, w) - _rolling_idxmin(low, w)) / w

        if _use("CORR"):
            for w in wlist:
                fields[f"CORR{w}"] = _rolling_corr(close, np.log(vol + 1), w)

        if _use("CORD"):
            ret = close / _ref(close, 1)
            vol_chg = np.log(vol / _ref(vol, 1) + 1)
            for w in wlist:
                fields[f"CORD{w}"] = _rolling_corr(ret, vol_chg, w)

        if _use("CNTP"):
            for w in wlist:
                fields[f"CNTP{w}"] = _rolling_mean((close > _ref(close, 1)).astype(float), w)

        if _use("CNTN"):
            for w in wlist:
                fields[f"CNTN{w}"] = _rolling_mean((close < _ref(close, 1)).astype(float), w)

        if _use("CNTD"):
            for w in wlist:
                up = _rolling_mean((close > _ref(close, 1)).astype(float), w)
                dn = _rolling_mean((close < _ref(close, 1)).astype(float), w)
                fields[f"CNTD{w}"] = up - dn

        if _use("SUMP"):
            delta = close - _ref(close, 1)
            for w in wlist:
                fields[f"SUMP{w}"] = _rolling_mean(np.maximum(delta, 0), w) / (
                    _rolling_mean(delta.abs(), w) + 1e-12
                )

        if _use("SUMN"):
            delta = close - _ref(close, 1)
            for w in wlist:
                fields[f"SUMN{w}"] = _rolling_mean(np.maximum(-delta, 0), w) / (
                    _rolling_mean(delta.abs(), w) + 1e-12
                )

        if _use("SUMD"):
            delta = close - _ref(close, 1)
            for w in wlist:
                up_ = _rolling_mean(np.maximum(delta, 0), w)
                dn_ = _rolling_mean(np.maximum(-delta, 0), w)
                fields[f"SUMD{w}"] = (up_ - dn_) / (_rolling_mean(delta.abs(), w) + 1e-12)

        if _use("VMA"):
            for w in wlist:
                fields[f"VMA{w}"] = _rolling_mean(vol, w) / (vol + 1e-12)

        if _use("VSTD"):
            for w in wlist:
                fields[f"VSTD{w}"] = _rolling_std(vol, w) / (vol + 1e-12)

        if _use("WVMA"):
            weighted = (close / _ref(close, 1) - 1).abs() * vol
            for w in wlist:
                fields[f"WVMA{w}"] = _rolling_std(weighted, w) / (
                    _rolling_mean(weighted, w) + 1e-12
                )

        if _use("VSUMP"):
            v_delta = vol - _ref(vol, 1)
            for w in wlist:
                fields[f"VSUMP{w}"] = _rolling_mean(np.maximum(v_delta, 0), w) / (
                    _rolling_mean(v_delta.abs(), w) + 1e-12
                )

        if _use("VSUMN"):
            v_delta = vol - _ref(vol, 1)
            for w in wlist:
                fields[f"VSUMN{w}"] = _rolling_mean(np.maximum(-v_delta, 0), w) / (
                    _rolling_mean(v_delta.abs(), w) + 1e-12
                )

        if _use("VSUMD"):
            v_delta = vol - _ref(vol, 1)
            for w in wlist:
                up_ = _rolling_mean(np.maximum(v_delta, 0), w)
                dn_ = _rolling_mean(np.maximum(-v_delta, 0), w)
                fields[f"VSUMD{w}"] = (up_ - dn_) / (_rolling_mean(v_delta.abs(), w) + 1e-12)

        return pd.DataFrame(fields, index=self._df.index)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def build_alpha360(df: pd.DataFrame) -> pd.DataFrame:
    """One-liner: build Alpha360 features from *df*."""
    return AlphaFactorBuilder(df).build_alpha360()


def build_alpha158(
    df: pd.DataFrame,
    kbar: bool = True,
    price_windows: Optional[List[int]] = None,
    volume_windows: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    rolling_include: Optional[List[str]] = None,
    rolling_exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One-liner: build Alpha158 features from *df*."""
    return AlphaFactorBuilder(df).build_alpha158(
        kbar=kbar,
        price_windows=price_windows,
        volume_windows=volume_windows,
        rolling_windows=rolling_windows,
        rolling_include=rolling_include,
        rolling_exclude=rolling_exclude,
    )
