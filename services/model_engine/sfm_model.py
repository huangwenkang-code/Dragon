"""
State Frequency Memory (SFM) model adapted from Microsoft Qlib.

SFM models time-series dynamics in the frequency domain.
Unlike standard RNNs, SFM decomposes hidden states into frequency
components, allowing the model to capture patterns at different
temporal scales.

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
import torch.nn.init as init
import torch.optim as optim

from .base import BaseModel, ArrayLike, SeriesLike
from .pytorch_utils import count_parameters, get_or_create_path

logger = logging.getLogger(__name__)


# =============================================================================
# SFM network module
# =============================================================================

class SFMNet(nn.Module):
    """State Frequency Memory recurrent cell.

    Parameters
    ----------
    d_feat : int
        Input feature dimension per time step.
    output_dim : int
        Dimension of the output projection before the final linear layer.
    freq_dim : int
        Number of frequency components.
    hidden_size : int
        Hidden state dimension.
    dropout_W : float
        Dropout on input weights.
    dropout_U : float
        Dropout on recurrent weights.
    device : str | torch.device
        Torch device.
    """

    def __init__(
        self,
        d_feat: int = 6,
        output_dim: int = 1,
        freq_dim: int = 10,
        hidden_size: int = 64,
        dropout_W: float = 0.0,
        dropout_U: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        # Input gate
        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        # State gate
        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        # Frequency gate
        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        # Candidate state
        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        # Output gate
        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        # Frequency -> hidden attention
        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        # Output projection
        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 1)

        self.states: List = []

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.reshape(len(input), self.input_dim, -1)  # [N, F, T]
        input = input.permute(0, 2, 1)                          # [N, T, F]
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't been initialized
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x: torch.Tensor) -> None:
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]

    def get_constants(self, x: torch.Tensor) -> None:
        constants: List = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))
        self.states[5:] = constants


# =============================================================================
# Trainer wrapper
# =============================================================================

class SFMModel(BaseModel):
    """State Frequency Memory training wrapper.

    Parameters
    ----------
    d_feat : int
        Input feature dimension per time step.
    hidden_size : int
        Hidden state dimension.
    output_dim : int
        Output projection dimension before the final linear layer.
    freq_dim : int
        Number of frequency components.
    dropout_W : float
        Dropout on input weights.
    dropout_U : float
        Dropout on recurrent weights.
    n_epochs : int
        Max training epochs.
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
        torch device.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        output_dim: int = 1,
        freq_dim: int = 10,
        dropout_W: float = 0.0,
        dropout_U: float = 0.0,
        n_epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "gd",
        device: Optional[Union[str, int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
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

        self._model = SFMNet(
            d_feat=self.d_feat,
            output_dim=self.output_dim,
            hidden_size=self.hidden_size,
            freq_dim=self.freq_dim,
            dropout_W=self.dropout_W,
            dropout_U=self.dropout_U,
            device=str(self.device),
        )
        logger.info("SFM model size: %.4f MB", count_parameters(self._model))

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
    def model(self) -> SFMNet:
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
    ) -> "SFMModel":
        """Train the SFM model.

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

        logger.info("Training SFM...")
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
            raise RuntimeError("SFMModel is not fitted yet. Call fit() first.")
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
