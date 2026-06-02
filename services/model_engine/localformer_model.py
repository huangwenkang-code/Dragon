"""
Localformer model adapted from Microsoft Qlib.

Localformer combines a Transformer encoder enhanced with 1D convolutions
(LocalformerEncoder) and a GRU for temporal aggregation. The convolutional
branch captures local temporal patterns, while the self-attention captures
long-range dependencies.

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.container import ModuleList

from .base import BaseModel, ArrayLike, SeriesLike
from .pytorch_utils import count_parameters, get_or_create_path

logger = logging.getLogger(__name__)


# =============================================================================
# Localformer network module
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


def _get_clones(module: nn.Module, N: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


class LocalformerEncoder(nn.Module):
    """Transformer encoder augmented with 1D convolutions for local context.

    Each layer computes: output = SelfAttention(output + Conv1D(output))
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, d_model: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers)
        self.num_layers = num_layers

    def forward(
        self, src: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = src
        out = src

        for i, mod in enumerate(self.layers):
            # [T, N, F] --> [N, T, F] --> [N, F, T]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)

            output = mod(output + out, src_mask=mask)

        return output + out


class LocalformerNet(nn.Module):
    """Localformer: Transformer + Conv1D + GRU hybrid.

    Parameters
    ----------
    d_feat : int
        Number of features per time step.
    d_model : int
        Hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dropout : float
        Dropout probability.
    device : str | torch.device | None
        Torch device.
    """

    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = LocalformerEncoder(
            self.encoder_layer, num_layers=num_layers, d_model=d_model
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F]
        src = src.transpose(1, 0)

        mask: Optional[torch.Tensor] = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, F] via last timestep -> linear -> scalar
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])

        return output.squeeze()


# =============================================================================
# Trainer wrapper
# =============================================================================

class LocalformerModel(BaseModel):
    """Localformer training wrapper for time-series regression.

    Parameters
    ----------
    d_feat : int
        Features per timestep.
    d_model : int
        Hidden dimension.
    nhead : int
        Attention heads.
    num_layers : int
        Number of encoder layers.
    dropout : float
        Dropout.
    n_epochs : int
        Max training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    early_stop : int
        Patience for early stopping.
    reg : float
        Weight decay (L2 regularization).
    loss : str
        Loss name (``"mse"``).
    optimizer : str
        ``"adam"`` or ``"gd"``.
    device : str | int | None
        torch device.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 100,
        lr: float = 0.0001,
        early_stop: int = 5,
        reg: float = 1e-3,
        loss: str = "mse",
        optimizer: str = "adam",
        device: Optional[Union[str, int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.loss_name = loss

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._model = LocalformerNet(
            d_feat=d_feat,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device,
        )
        logger.info("Localformer model size: %.4f MB", count_parameters(self._model))

        opt = optimizer.lower()
        if opt == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=reg)
        elif opt in ("gd", "sgd"):
            self._optimizer = optim.SGD(self._model.parameters(), lr=self.lr, weight_decay=reg)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

        self._model.to(self.device)
        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    @property
    def model(self) -> LocalformerNet:
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
        mask = torch.isfinite(label)
        return -self._loss_fn(pred[mask], label[mask])

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
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
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
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
    ) -> "LocalformerModel":
        """Train the Localformer model.

        Parameters
        ----------
        X, y : training data.
        X_valid, y_valid : validation data.
        save_path : str | None
            If provided, the best model state is saved to this path.
        """
        X_train = self._as_array(X)
        y_train = self._as_array(y).ravel()
        X_v = self._as_array(X_valid) if X_valid is not None else X_train
        y_v = self._as_array(y_valid).ravel() if y_valid is not None else y_train

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}

        logger.info("Training Localformer...")
        for epoch in range(self.n_epochs):
            logger.info("Epoch %d:", epoch)
            self._train_epoch(X_train, y_train)
            train_loss, train_score = self._test_epoch(X_train, y_train)
            val_loss, val_score = self._test_epoch(X_v, y_v)
            logger.info("train %.6f, valid %.6f", train_score, val_score)

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
            torch.save(best_state, get_or_create_path(save_path))

        if self.use_gpu:
            torch.cuda.empty_cache()
        self._fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return scalar predictions for each sample in *X*."""
        if not self._fitted:
            raise RuntimeError("LocalformerModel is not fitted yet. Call fit() first.")
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
