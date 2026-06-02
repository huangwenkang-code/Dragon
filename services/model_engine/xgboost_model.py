"""
XGBoost model wrapper adapted from Microsoft Qlib.

Extracts the core XGBoost training/inference logic from
qlib.contrib.model.xgboost and re-implements it against plain
numpy/pandas arrays (no qlib ``Dataset`` dependency).

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import xgboost as xgb

from .base import BaseModel, ArrayLike, SeriesLike


class XGBoostModel(BaseModel):
    """XGBoost wrapper with fit/predict interface.

    Parameters
    ----------
    **kwargs :
        Passed through to ``xgb.train`` as the ``params`` dict.
        Common keys: ``objective``, ``eval_metric``, ``max_depth``,
        ``learning_rate``, ``subsample``, ``colsample_bytree``,
        ``reg_alpha``, ``reg_lambda``, ``tree_method``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._params: Dict[str, Any] = dict(kwargs)
        self._model: Optional[xgb.Booster] = None
        self._feature_names: Optional[List[str]] = None
        self._evals_result: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: SeriesLike,
        *,
        X_valid: Optional[ArrayLike] = None,
        y_valid: Optional[SeriesLike] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 20,
        sample_weight: Optional[np.ndarray] = None,
        sample_weight_valid: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> XGBoostModel:
        """Train an XGBoost model.

        Parameters
        ----------
        X, y : training data.
        X_valid, y_valid : validation data (used for early stopping).
        num_boost_round : int
            Max number of boosting rounds.
        early_stopping_rounds : int
            Stop if validation metric does not improve for this many rounds.
        verbose_eval : int
            Log evaluation metric every N rounds (0 = silent).
        sample_weight : np.ndarray | None
            Instance weights for training set.
        sample_weight_valid : np.ndarray | None
            Instance weights for validation set.
        """
        X_arr = self._as_array(X)
        y_arr = self._as_array(y).ravel()  # XGBoost needs 1-d label

        # Capture feature names from DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        dtrain = xgb.DMatrix(X_arr, label=y_arr, weight=sample_weight,
                             feature_names=self._feature_names)

        # Build eval list
        evals: List[tuple] = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            Xv_arr = self._as_array(X_valid)
            yv_arr = self._as_array(y_valid).ravel()
            dvalid = xgb.DMatrix(Xv_arr, label=yv_arr, weight=sample_weight_valid,
                                 feature_names=self._feature_names)
            evals.append((dvalid, "valid"))

        evals_result: Dict[str, List[float]] = {}
        self._model = xgb.train(
            self._params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )

        # Flatten evals_result keys (xgb nests them by metric name)
        self._evals_result = {}
        for split in evals_result:
            if evals_result[split]:
                key = list(evals_result[split].keys())[0]
                self._evals_result[split] = evals_result[split][key]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return raw prediction values (scores)."""
        if self._model is None:
            raise RuntimeError("XGBoostModel is not fitted yet. Call fit() first.")
        X_arr = self._as_array(X)
        dtest = xgb.DMatrix(X_arr, feature_names=self._feature_names)
        return self._model.predict(dtest)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (loss per round for each eval set)."""
        return self._evals_result

    @property
    def booster(self) -> xgb.Booster:
        """Return the underlying ``xgboost.Booster``."""
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        return self._model

    def get_feature_importance(
        self,
        importance_type: str = "weight",
    ) -> pd.Series:
        """Return feature importance, sorted descending.

        Parameters
        ----------
        importance_type : str
            ``weight``, ``gain``, ``cover``, ``total_gain``, or ``total_cover``.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        scores = self._model.get_score(importance_type=importance_type)
        return pd.Series(scores).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Persist model to disk."""
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> XGBoostModel:
        """Load a persisted model from disk."""
        inst = cls()
        inst._model = xgb.Booster()
        inst._model.load_model(path)
        inst._fitted = True
        return inst
