"""
Base model interface adapted from Microsoft Qlib.

The original qlib.model.base defines ``Model`` with a ``fit(dataset)``
signature that is tightly coupled to ``qlib.data.dataset.Dataset``.
This module provides a simplified equivalent that works directly with
pandas DataFrames and numpy arrays -- no qlib runtime dependency.

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]
SeriesLike = Union[np.ndarray, pd.Series]


class BaseModel(abc.ABC):
    """Abstract base for all learnable models in dragon-engine.

    Subclasses must implement ``fit`` and ``predict``.

    Usage::

        model = MyModel(...)
        model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
        preds = model.predict(X_test)
    """

    def __init__(self, **kwargs: Any) -> None:
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True after a successful ``fit`` call."""
        return self._fitted

    def _as_array(self, data: ArrayLike) -> np.ndarray:
        """Convert *data* to a plain float32 numpy array."""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values.astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """Syntactic sugar: ``model(X)`` delegates to ``model.predict(X)``."""
        return self.predict(X)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: SeriesLike,
        *,
        X_valid: Optional[ArrayLike] = None,
        y_valid: Optional[SeriesLike] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Train the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Training labels.
        X_valid : array-like, optional
            Validation features.
        y_valid : array-like, optional
            Validation labels.
        kwargs : dict
            Additional model-specific parameters.

        Returns
        -------
        self : BaseModel
        """
        ...

    @abc.abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        preds : np.ndarray, shape (n_samples,)
        """
        ...
