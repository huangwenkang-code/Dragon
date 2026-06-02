"""
Transformer model adapted from Microsoft Qlib.

Extracts the Transformer architecture and training loop from
qlib.contrib.model.pytorch_transformer, re-implemented to run
standalone with plain numpy/pandas inputs (no qlib dependency).

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

import copy
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseModel, ArrayLike, SeriesLike

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch modules
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in "Attention Is All You Need"."""

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, N, F]"""
        return x + self.pe[: x.size(0), :]


class TransformerModule(nn.Module):
    """Transformer encoder that processes temporal feature sequences.

    Expects input shaped ``[N, F * T]`` (flattened multi-timestep features),
    internally reshapes to ``[T, N, F]`` for the encoder.

    Parameters
    ----------
    d_feat : int
        Number of features per time step.
    d_model : int
        Hidden dimension of the transformer.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        src : [N, F * T]
            Flattened multi-timestep features (e.g. 60 days x 6 fields = 360).

        Returns
        -------
        out : [N]
            Scalar prediction per sample.
        """
        # [N, F*T] -> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # [N, T, F] -> [T, N, F]  (not batch-first)
        src = src.transpose(1, 0)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # Take the last time step -> [N, d_model] -> [N, 1]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])
        return output.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TransformerModel(BaseModel):
    """Training wrapper around the Transformer module.

    Parameters
    ----------
    d_feat : int
        Features per timestep in the raw input.
    d_model : int
        Transformer hidden dimension.
    nhead : int
        Attention heads.
    num_layers : int
        Encoder layers.
    dropout : float
        Dropout rate.
    batch_size : int
        Training batch size.
    n_epochs : int
        Max training epochs.
    lr : float
        Learning rate.
    early_stop : int
        Patience for early stopping.
    loss : str
        Loss name (currently only ``"mse"``).
    optimizer : str
        ``"adam"`` or ``"sgd"``.
    weight_decay : float
        L2 regularisation.
    device : str | None
        ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or None for auto-detect.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
        batch_size: int = 2048,
        n_epochs: int = 100,
        lr: float = 1e-4,
        early_stop: int = 5,
        loss: str = "mse",
        optimizer: str = "adam",
        weight_decay: float = 1e-3,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.loss_name = loss
        self.weight_decay = weight_decay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._model = TransformerModule(d_feat, d_model, nhead, num_layers, dropout)
        opt = optimizer.lower()
        if opt == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt in ("sgd", "gd"):
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

        self._model.to(self.device)
        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    @property
    def model(self) -> TransformerModule:
        return self._model

    @property
    def use_gpu(self) -> bool:
        return self.device.type != "cpu"

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _loss_fn(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(label)
        if self.loss_name == "mse":
            return torch.mean((pred[mask].float() - label[mask].float()) ** 2)
        raise ValueError(f"Unknown loss: {self.loss_name}")

    def _metric_fn(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Returns a score where higher = better (negated loss)."""
        return -self._loss_fn(pred, label)

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)

            pred = self._model(feat)
            loss = self._loss_fn(pred, lbl)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self._optimizer.step()

    def _test_epoch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self._model.eval()
        losses: List[float] = []
        scores: List[float] = []

        indices = np.arange(len(X))
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)

            with torch.no_grad():
                pred = self._model(feat)
                loss = self._loss_fn(pred, lbl)
                losses.append(loss.item())
                scores.append(self._metric_fn(pred, lbl).item())

        return float(np.mean(losses)), float(np.mean(scores))

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: SeriesLike,
        *,
        X_valid: Optional[ArrayLike] = None,
        y_valid: Optional[SeriesLike] = None,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> TransformerModel:
        """Train the transformer.

        Requires *X* and *y* as numpy arrays.  Validation data is optional
        (if omitted, early-stopping is done on training loss, which is usually
        not what you want -- always provide validation data when possible).
        """
        X_train = self._as_array(X)
        y_train = self._as_array(y).ravel()
        X_v = self._as_array(X_valid) if X_valid is not None else X_train
        y_v = self._as_array(y_valid).ravel() if y_valid is not None else y_train

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}

        for epoch in range(self.n_epochs):
            self._train_epoch(X_train, y_train)
            train_loss, train_score = self._test_epoch(X_train, y_train)
            val_loss, val_score = self._test_epoch(X_v, y_v)

            self._evals_result["train"].append(train_score)
            self._evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_state = copy.deepcopy(self._model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._model.load_state_dict(best_state)
        if save_path is not None:
            torch.save(best_state, save_path)
        self._fitted = True

        if self.use_gpu:
            torch.cuda.empty_cache()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return scalar predictions for each sample in *X*."""
        if not self._fitted:
            raise RuntimeError("TransformerModel is not fitted yet. Call fit() first.")
        X_arr = self._as_array(X)
        self._model.eval()
        preds: List[np.ndarray] = []
        n = len(X_arr)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.from_numpy(X_arr[start:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self._model(batch).detach().cpu().numpy())

        return np.concatenate(preds)

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
