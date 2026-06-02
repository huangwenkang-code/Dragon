"""
Model ensemble orchestrator for leader stock scoring.

Combines predictions from multiple model types:
  - XGBoost: tabular factor scoring (fast, calibrated)
  - GATs: cross-stock graph attention (leader-follower relationships)
  - Transformer: temporal sequence modeling
  - TCN / LSTM / GRU: alternative temporal architectures

Each model produces a per-stock score; the ensemble fuses them
with configurable weights into the final leader_score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger(__name__)


class ModelEnsemble:
    """Weighted ensemble of stock scoring models.

    Usage::

        ensemble = ModelEnsemble(weights={
            "xgboost": 0.35,
            "gats": 0.30,
            "transformer": 0.20,
            "tcn": 0.15,
        })
        scores = ensemble.predict_all(features, graph_data)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Model name -> weight dict. Normalized to sum=1.
                     Default weights balance tabular, graph, and temporal.
        """
        self.weights = weights or {
            "xgboost": 0.30,
            "gats": 0.25,
            "transformer": 0.20,
            "tcn": 0.15,
            "lstm": 0.10,
        }
        self._normalize_weights()
        self._models: Dict[str, Any] = {}
        self._fitted = False

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def add_model(self, name: str, model: Any) -> None:
        """Register a fitted model under a name matching a weight key."""
        self._models[name] = model
        logger.info("Ensemble: registered %s (total models: %d)", name, len(self._models))

    def remove_model(self, name: str) -> None:
        self._models.pop(name, None)
        # Re-normalize excluding removed model
        self.weights.pop(name, None)
        self._normalize_weights()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_single(
        self,
        features: pd.DataFrame,
        graph_data: Optional[Any] = None,
    ) -> pd.Series:
        """Return ensemble score for each stock (0-1 scale).

        Args:
            features: DataFrame indexed by stock code, columns = factor features.
            graph_data: Optional graph structure for GATs model.

        Returns:
            Series indexed by stock code with ensemble scores.
        """
        if not self._models:
            logger.warning("Ensemble has no models — returning zero scores")
            return pd.Series(0.0, index=features.index, name="ensemble_score")

        all_scores: Dict[str, pd.Series] = {}

        for name, model in self._models.items():
            if name not in self.weights:
                continue
            try:
                score = self._predict_one(name, model, features, graph_data)
                if score is not None:
                    all_scores[name] = score
            except Exception as exc:
                logger.warning("Ensemble: %s prediction failed: %s", name, exc)

        if not all_scores:
            return pd.Series(0.0, index=features.index, name="ensemble_score")

        return self._fuse(all_scores, features.index)

    def _predict_one(
        self,
        name: str,
        model: Any,
        features: pd.DataFrame,
        graph_data: Optional[Any],
    ) -> Optional[pd.Series]:
        """Route prediction to the right model type."""
        # Tabular models (XGBoost, CatBoost, TabNet)
        if name in ("xgboost", "catboost", "tabnet", "lightgbm"):
            if hasattr(model, "predict"):
                raw = model.predict(features)
                return self._to_series(raw, features.index, name)

        # Graph models (GATs)
        if name == "gats" and graph_data is not None:
            if hasattr(model, "predict"):
                raw = model.predict(features, graph_data)
                return self._to_series(raw, features.index, name)

        # Temporal models (Transformer, TCN, LSTM, GRU, ALSTM, SFM)
        if name in ("transformer", "tcn", "lstm", "gru", "alstm", "sfm", "tra", "hist", "localformer", "adarnn"):
            if hasattr(model, "predict"):
                raw = model.predict(features)
                return self._to_series(raw, features.index, name)

        return None

    @staticmethod
    def _to_series(raw: Any, index: pd.Index, name: str) -> pd.Series:
        """Convert raw prediction to a normalized Series."""
        if isinstance(raw, pd.Series):
            return raw
        if isinstance(raw, np.ndarray):
            raw = raw.flatten()
        ser = pd.Series(raw, index=index[:len(raw)], name=name)
        # Normalize to [0, 1]
        ser = (ser - ser.min()) / (ser.max() - ser.min() + 1e-8)
        return ser

    def _fuse(self, all_scores: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
        """Weighted fusion of per-model scores."""
        # Re-normalize weights for available models only
        available_weights = {k: v for k, v in self.weights.items() if k in all_scores}
        w_total = sum(available_weights.values()) or 1.0
        norm_weights = {k: v / w_total for k, v in available_weights.items()}

        fused = pd.Series(0.0, index=index, name="ensemble_score")
        for name, score in all_scores.items():
            w = norm_weights.get(name, 0.0)
            fused = fused.add(score * w, fill_value=0.0)

        return fused.clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Feature importance (from XGBoost)
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_k: int = 20) -> Dict[str, Any]:
        """Get feature importance from the XGBoost model if available."""
        xgb = self._models.get("xgboost")
        if xgb is None:
            return {"error": "XGBoost not in ensemble"}

        if hasattr(xgb, "get_feature_importance"):
            imp = xgb.get_feature_importance()
            return {
                "top_features": imp.head(top_k).to_dict(),
                "total_features": len(imp),
            }

        if hasattr(xgb, "feature_importances_"):
            imp = pd.Series(xgb.feature_importances_)
            imp.sort_values(ascending=False, inplace=True)
            return {
                "top_features": imp.head(top_k).to_dict(),
                "total_features": len(imp),
            }

        return {"error": "No feature importance available"}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "model_names": list(self._models.keys()),
            "fitted": self._fitted,
        }

    def set_fitted(self, fitted: bool = True) -> None:
        self._fitted = fitted

    @property
    def is_fitted(self) -> bool:
        return self._fitted and len(self._models) > 0

    @property
    def model_count(self) -> int:
        return len(self._models)
