"""
Graph Attention Network (GATs) model adapted from Microsoft Qlib.

Extracts the GATs network architecture from
qlib.contrib.model.pytorch_gats — a relational attention mechanism
on top of an RNN backbone that is designed to capture inter-stock
relationships.

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
import torch.optim as optim

from .base import BaseModel, ArrayLike, SeriesLike

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# GAT Module:  RNN backbone + graph attention on final hidden states
# ═══════════════════════════════════════════════════════════════════════════

class GATModule(nn.Module):
    """Graph Attention network on top of an RNN sequence encoder.

    For a batch of N stocks on a single day, each stock is encoded through the
    RNN over its temporal features.  The final hidden states of all N stocks
    then attend to each other via a learned graph-attention mechanism, allowing
    the model to capture cross-sectional stock relationships.

    Parameters
    ----------
    d_feat : int
        Number of features per time step.
    hidden_size : int
        RNN hidden size (also the attention dimension).
    num_layers : int
        Number of recurrent layers.
    dropout : float
        Dropout applied between RNN layers.
    base_model : str
        Recurrent cell: ``"GRU"`` (default) or ``"LSTM"``.
    """

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        base_model: str = "GRU",
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        # -- recurrent backbone ------------------------------------------
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}.get(base_model.upper())
        if rnn_cls is None:
            raise ValueError(f"Unknown base_model '{base_model}'. Choose 'GRU' or 'LSTM'.")
        self.rnn: nn.RNNBase = rnn_cls(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # -- graph attention layers --------------------------------------
        self.transformation = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.randn(hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def _cal_attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise graph-attention weights between *x* and *y*.

        Parameters
        ----------
        x, y : [N, H]

        Returns
        -------
        att_weight : [N, N]
        """
        x = self.transformation(x)
        y = self.transformation(y)
        N, H = x.shape
        e_x = x.expand(N, N, H)          # [N, N, H]
        e_y = e_x.transpose(0, 1)        # [N, N, H]
        attention_in = torch.cat((e_x, e_y), dim=2).reshape(-1, H * 2)  # [N*N, 2H]
        attention_out = self.a.t().mm(attention_in.t())  # [1, N*N]
        attention_out = attention_out.view(N, N)
        attention_out = self.leaky_relu(attention_out)
        return self.softmax(attention_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, F * T]
            Flattened multi-timestep features for N stocks on one day.

        Returns
        -------
        out : [N]
            Scalar prediction per stock.
        """
        # [N, F*T] -> [N, T, F]
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        out, _ = self.rnn(x)             # out: [N, T, H]
        hidden = out[:, -1, :]           # [N, H]  -- last time step

        att_weight = self._cal_attention(hidden, hidden)  # [N, N]
        hidden = att_weight.mm(hidden) + hidden           # residual
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze(-1)            # [N]


# ═══════════════════════════════════════════════════════════════════════════
# Trainer wrapper
# ═══════════════════════════════════════════════════════════════════════════

class GATsModel(BaseModel):
    """Training wrapper for the GATs architecture.

    Note on batching:
        Unlike the Transformer which uses random mini-batches, GATs
        batches *per day*.  All stocks belonging to the same day are
        processed together so the graph attention sees the full
        cross-section.  Your input DataFrames **must** have a
        ``MultiIndex`` of ``(datetime, instrument)`` or, if a plain
        numpy array, consecutive rows must correspond to the same day.

    Parameters
    ----------
    d_feat : int
        Features per timestep.
    hidden_size : int
        RNN/attention hidden dimension.
    num_layers : int
        RNN layers.
    dropout : float
        Dropout.
    base_model : str
        ``"GRU"`` or ``"LSTM"``.
    n_epochs : int
        Max epochs.
    lr : float
        Learning rate.
    early_stop : int
        Patience for early stopping.
    loss : str
        Loss name (``"mse"``).
    optimizer : str
        ``"adam"`` or ``"sgd"``.
    device : str | None
        torch device.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        base_model: str = "GRU",
        n_epochs: int = 200,
        lr: float = 1e-3,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.loss_name = loss

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._model = GATModule(d_feat, hidden_size, num_layers, dropout, base_model)
        logger.info("GATs model size: %.4f MB", _count_parameters(self._model) * 4 / 1024 / 1024)

        opt = optimizer.lower()
        if opt == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif opt in ("sgd", "gd"):
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

        self._model.to(self.device)
        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    @property
    def model(self) -> GATModule:
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
        return -self._loss_fn(pred, label)

    # ------------------------------------------------------------------
    # Daily batching
    # ------------------------------------------------------------------

    @staticmethod
    def _get_daily_slices(
        df: pd.DataFrame, shuffle: bool = False
    ) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` slices for each day in *df*.

        *df* must have a MultiIndex (datetime, instrument).
        """
        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
            raise ValueError(
                "GATsModel expects a DataFrame with MultiIndex (datetime, instrument)"
            )
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        slices = list(zip(daily_index, daily_index + daily_count))
        if shuffle:
            np.random.shuffle(slices)
        return [(int(s), int(e)) for s, e in slices]

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> None:
        self._model.train()
        slices = self._get_daily_slices(df, shuffle=True)
        for start, end in slices:
            feat = torch.from_numpy(X[start:end]).float().to(self.device)
            lbl = torch.from_numpy(y[start:end]).float().to(self.device)

            pred = self._model(feat)
            loss = self._loss_fn(pred, lbl)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self._optimizer.step()

    def _test_epoch(
        self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame
    ) -> Tuple[float, float]:
        self._model.eval()
        losses: List[float] = []
        scores: List[float] = []
        slices = self._get_daily_slices(df, shuffle=False)

        for start, end in slices:
            feat = torch.from_numpy(X[start:end]).float().to(self.device)
            lbl = torch.from_numpy(y[start:end]).float().to(self.device)

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
    ) -> GATsModel:
        """Train the GATs model.

        *X* and *X_valid* must be pandas DataFrames with a (datetime, instrument)
        MultiIndex so daily batching can be inferred.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GATsModel.fit() requires X as a pd.DataFrame with MultiIndex")
        X_train = X
        y_train_vals = self._as_array(y).ravel()
        X_train_arr = self._as_array(X)

        # Use training index if no validation provided
        if X_valid is None or y_valid is None:
            X_val = X_train
            y_val_vals = y_train_vals
            X_val_arr = X_train_arr
        else:
            if not isinstance(X_valid, pd.DataFrame):
                raise TypeError("GATsModel.fit() requires X_valid as a pd.DataFrame")
            X_val = X_valid
            y_val_vals = self._as_array(y_valid).ravel()
            X_val_arr = self._as_array(X_valid)

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}

        for epoch in range(self.n_epochs):
            self._train_epoch(X_train_arr, y_train_vals, X_train)
            train_loss, train_score = self._test_epoch(X_train_arr, y_train_vals, X_train)
            val_loss, val_score = self._test_epoch(X_val_arr, y_val_vals, X_val)

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
        """Return scalar predictions.

        *X* should be a pandas DataFrame with (datetime, instrument)
        MultiIndex if you want the results to be correctly ordered by
        day.  Plain numpy arrays are processed in a single batch.
        """
        if not self._fitted:
            raise RuntimeError("GATsModel is not fitted yet. Call fit() first.")

        X_arr = self._as_array(X)
        self._model.eval()
        preds: List[np.ndarray] = []

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
            slices = self._get_daily_slices(X, shuffle=False)
            for start, end in slices:
                batch = torch.from_numpy(X_arr[start:end]).float().to(self.device)
                with torch.no_grad():
                    preds.append(self._model(batch).detach().cpu().numpy())
        else:
            # Single batch fallback
            batch = torch.from_numpy(X_arr).float().to(self.device)
            with torch.no_grad():
                preds.append(self._model(batch).detach().cpu().numpy())

        return np.concatenate(preds)

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
