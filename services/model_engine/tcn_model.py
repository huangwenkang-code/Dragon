"""
Temporal Convolutional Network (TCN) model adapted from Microsoft Qlib.

The TCN architecture uses dilated causal convolutions to capture
long-range temporal dependencies in financial time series.

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim

from .base import BaseModel, ArrayLike, SeriesLike
from .pytorch_utils import count_parameters, get_or_create_path

logger = logging.getLogger(__name__)


# =============================================================================
# TCN building blocks (from qlib/contrib/model/tcn.py)
# =============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# TCN network module
# =============================================================================

class TCNNet(nn.Module):
    """TCN backbone: dilated causal convs + linear projection.

    Parameters
    ----------
    num_input : int
        Number of features per timestep.
    output_size : int
        Output dimension (1 for regression).
    num_channels : list[int]
        Number of channels per TCN layer.
    kernel_size : int
        Convolution kernel size.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_input: int = 6,
        output_size: int = 1,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [128, 128, 128, 128, 128]
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, F*T]
        x = x.reshape(x.shape[0], self.num_input, -1)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()


# =============================================================================
# Trainer wrapper
# =============================================================================

class TCNModel(BaseModel):
    """Temporal Convolutional Network for time-series forecasting.

    Parameters
    ----------
    d_feat : int
        Input dimension for each time step.
    n_chans : int
        Number of channels per TCN layer.
    kernel_size : int
        Convolution kernel size.
    num_layers : int
        Number of TCN layers (stacked dilated convolutions).
    dropout : float
        Dropout probability.
    n_epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    early_stop : int
        Patience for early stopping.
    loss : str
        Loss name (``"mse"``).
    optimizer : str
        ``"adam"`` or ``"gd"``.
    device : str | int | None
        ``"cuda"``, ``"cuda:0"``, ``"cpu"``, GPU ID, or None for auto-detect.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        d_feat: int = 6,
        n_chans: int = 128,
        kernel_size: int = 5,
        num_layers: int = 5,
        dropout: float = 0.5,
        n_epochs: int = 200,
        lr: float = 0.0001,
        batch_size: int = 2000,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        device: Optional[Union[str, int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
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

        self._model = TCNNet(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        logger.info("TCN model size: %.4f MB", count_parameters(self._model))

        opt = optimizer.lower()
        if opt == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        elif opt in ("gd", "sgd"):
            self._optimizer = optim.SGD(self._model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

        self._model.to(self.device)
        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    @property
    def model(self) -> TCNNet:
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
            return torch.mean((pred[mask] - label[mask]) ** 2)
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
    ) -> "TCNModel":
        """Train the TCN model.

        Parameters
        ----------
        X, y : training data.
        X_valid, y_valid : validation data (used for early stopping).
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

        logger.info("Training TCN...")
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
            raise RuntimeError("TCNModel is not fitted yet. Call fit() first.")
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
