"""
HIST (Hierarchical Stock Transformer) model adapted from Microsoft Qlib.

HIST uses concept-based information sharing among stocks. Given a
predefined stock-to-concept mapping (e.g., industry sectors), it learns
three types of information:
1. **Predefined Concept** -- shared information from known concepts
2. **Hidden Concept** -- dynamically discovered inter-stock relations
3. **Individual Information** -- stock-specific patterns

The three components are summed for the final prediction.

Reference: Xu et al., "HIST: A Graph-based Framework for Stock Trend
Forecasting via Mining Concept-Oriented Shared Information" (2021).

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
from .pytorch_utils import count_parameters, get_or_create_path

logger = logging.getLogger(__name__)


# =============================================================================
# HIST network module
# =============================================================================

class HISTNet(nn.Module):
    """Hierarchical Stock Transformer network.

    Parameters
    ----------
    d_feat : int
        Features per time step.
    hidden_size : int
        Hidden state dimension.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout probability.
    base_model : str
        ``"GRU"`` or ``"LSTM"``.
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

        if base_model == "GRU":
            self.rnn: nn.RNNBase = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown base_model '{base_model}'")

        # Predefined Concept Module (external shared info)
        self.fc_es = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es.weight)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is.weight)

        self.fc_es_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_middle.weight)
        self.fc_is_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_middle.weight)

        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)
        self.fc_indi_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi_fore.weight)

        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)

        self.fc_out_es = nn.Linear(hidden_size, 1)
        self.fc_out_is = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def cal_cos_similarity(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
        return cos_similarity

    def forward(self, x: torch.Tensor, concept_matrix: torch.Tensor) -> torch.Tensor:
        device = x.device

        x_hidden = x.reshape(len(x), self.d_feat, -1)   # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)             # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]                    # [N, H]

        # ---------- Predefined Concept Module ----------
        stock_to_concept = concept_matrix

        stock_to_concept_sum = (
            torch.sum(stock_to_concept, 0)
            .reshape(1, -1)
            .repeat(stock_to_concept.shape[0], 1)
        )
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + torch.ones(
            stock_to_concept.shape[0], stock_to_concept.shape[1]
        ).to(device)
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        hidden = hidden[hidden.sum(1) != 0]

        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        e_shared_info = concept_to_stock.mm(hidden)
        e_shared_info = self.fc_es(e_shared_info)

        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.fc_es_fore(e_shared_info)
        output_es = self.leaky_relu(output_es)

        # ---------- Hidden Concept Module ----------
        i_shared_info = x_hidden - e_shared_back
        hidden = i_shared_info
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, hidden)
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        i_stock_to_concept = i_stock_to_concept * (
            torch.ones(dim, dim) - torch.eye(dim)
        ).to(device)
        row = torch.linspace(0, dim - 1, dim).to(device).long()
        column = i_stock_to_concept.max(1)[1].long()
        value = i_stock_to_concept.max(1)[0]
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed(
            (i_stock_to_concept.sum(0) != 0).float() * diag
        )
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
        hidden = hidden[hidden.sum(1) != 0]

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = i_concept_to_stock.mm(hidden)
        i_shared_info = self.fc_is(i_shared_info)

        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.fc_is_fore(i_shared_info)
        output_is = self.leaky_relu(output_is)

        # ---------- Individual Information Module ----------
        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)

        # ---------- Stock Trend Prediction ----------
        all_info = output_es + output_is + output_indi
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all


# =============================================================================
# Trainer wrapper
# =============================================================================

class HISTModel(BaseModel):
    """HIST (Hierarchical Stock Transformer) training wrapper.

    This model requires a **stock-to-concept matrix** and a **stock index
    mapping** to be provided at construction time. The concept matrix is
    an [N_stocks, N_concepts] binary/weighted matrix encoding industry or
    other categorical relationships.

    .. note::
        The original Qlib implementation downloads these files from a
        GitHub release if they are not found locally. This adapter
        requires them to be provided explicitly.

    Parameters
    ----------
    d_feat : int
        Features per timestep.
    hidden_size : int
        Hidden dimension.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout.
    base_model : str
        ``"GRU"`` or ``"LSTM"``.
    n_epochs : int
        Max training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size for training (daily batching is used internally).
    early_stop : int
        Patience for early stopping.
    loss : str
        Loss name (``"mse"``).
    optimizer : str
        ``"adam"`` or ``"gd"``.
    stock2concept : str
        Path to a ``.npy`` file containing the stock-to-concept matrix
        of shape (N_stocks, N_concepts).
    stock_index : str
        Path to a ``.npy`` file containing a dict mapping instrument
        codes to row indices in the concept matrix.
    device : str | int | None
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
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        stock2concept: Optional[str] = None,
        stock_index: Optional[str] = None,
        device: Optional[Union[str, int]] = None,
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
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_name = loss

        # Concept matrix paths
        # TODO: adapt to dragon-engine Dataset interface for concept matrix loading
        self.stock2concept_path = stock2concept
        self.stock_index_path = stock_index
        self._concept_matrix: Optional[np.ndarray] = None
        self._stock_index_map: Optional[Dict[Any, int]] = None

        # Load concept data if paths provided
        if stock2concept is not None:
            self._concept_matrix = np.load(stock2concept)
        if stock_index is not None:
            self._stock_index_map = np.load(stock_index, allow_pickle=True).item()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._model = HISTNet(d_feat, hidden_size, num_layers, dropout, base_model)
        logger.info("HIST model size: %.4f MB", count_parameters(self._model))

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
    def model(self) -> HISTNet:
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
    # Daily batching helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_daily_slices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return daily_index and daily_count arrays for *df*.

        *df* must have a ``MultiIndex`` of ``(datetime, instrument)``.
        """
        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
            raise ValueError("HISTModel expects a DataFrame with MultiIndex (datetime, instrument)")
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        return daily_index, daily_count

    def _get_concept_slices(
        self, df: pd.DataFrame, default_idx: int = 733
    ) -> np.ndarray:
        """Map each row in *df* to a concept-matrix row index."""
        if self._stock_index_map is None:
            # No stock index map -- return default index for all rows
            return np.full(len(df), default_idx, dtype=np.int64)

        codes = df.index.get_level_values("instrument")
        indices = np.array([self._stock_index_map.get(c, default_idx) for c in codes])
        return indices

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame
    ) -> None:
        if self._concept_matrix is None:
            raise RuntimeError(
                "Concept matrix is not loaded. Provide stock2concept path in constructor."
            )
        concept = self._concept_matrix
        stock_idx = self._get_concept_slices(df)

        self._model.train()
        daily_index, daily_count = self._get_daily_slices(df)
        # Shuffle daily order
        perm = np.random.permutation(len(daily_index))
        daily_index = daily_index[perm]
        daily_count = daily_count[perm]

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feat = torch.from_numpy(X[batch]).float().to(self.device)
            c_mat = torch.from_numpy(concept[stock_idx[batch]]).float().to(self.device)
            lbl = torch.from_numpy(y[batch]).float().to(self.device)

            pred = self._model(feat, c_mat)
            loss = self._loss_fn(pred, lbl)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self._optimizer.step()

    def _test_epoch(
        self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame
    ) -> Tuple[float, float]:
        if self._concept_matrix is None:
            raise RuntimeError("Concept matrix is not loaded.")
        concept = self._concept_matrix
        stock_idx = self._get_concept_slices(df)

        self._model.eval()
        losses: List[float] = []
        scores: List[float] = []

        daily_index, daily_count = self._get_daily_slices(df)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feat = torch.from_numpy(X[batch]).float().to(self.device)
            c_mat = torch.from_numpy(concept[stock_idx[batch]]).float().to(self.device)
            lbl = torch.from_numpy(y[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self._model(feat, c_mat)
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
    ) -> "HISTModel":
        """Train the HIST model.

        *X* should be a pandas DataFrame with a ``(datetime, instrument)``
        MultiIndex so daily batching can be inferred. The concept matrix
        must have been provided in the constructor.

        Parameters
        ----------
        X, y : training data (X must be a DataFrame with MultiIndex).
        X_valid, y_valid : validation data.
        save_path : str | None
            If provided, the best model state is saved to this path.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HISTModel.fit() requires X as a pd.DataFrame with MultiIndex")
        if X_valid is not None and not isinstance(X_valid, pd.DataFrame):
            raise TypeError("HISTModel.fit() requires X_valid as a pd.DataFrame")

        X_train_df = X
        y_train_vals = self._as_array(y).ravel()
        X_train_arr = self._as_array(X)

        if X_valid is None or y_valid is None:
            X_val_df = X_train_df
            y_val_vals = y_train_vals
            X_val_arr = X_train_arr
        else:
            X_val_df = X_valid
            y_val_vals = self._as_array(y_valid).ravel()
            X_val_arr = self._as_array(X_valid)

        # TODO: adapt to dragon-engine Dataset interface for loading
        # pretrained base_model weights via self.model_path

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}

        logger.info("Training HIST...")
        for epoch in range(self.n_epochs):
            logger.info("Epoch %d:", epoch)
            self._train_epoch(X_train_arr, y_train_vals, X_train_df)
            train_loss, train_score = self._test_epoch(X_train_arr, y_train_vals, X_train_df)
            val_loss, val_score = self._test_epoch(X_val_arr, y_val_vals, X_val_df)
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
        """Return scalar predictions.

        *X* should be a pandas DataFrame with ``(datetime, instrument)``
        MultiIndex for proper daily batching.
        """
        if not self._fitted:
            raise RuntimeError("HISTModel is not fitted yet. Call fit() first.")
        if self._concept_matrix is None:
            raise RuntimeError("Concept matrix is not loaded.")

        X_arr = self._as_array(X)
        self._model.eval()
        concept = self._concept_matrix

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
            stock_idx = self._get_concept_slices(X)
            daily_index, daily_count = self._get_daily_slices(X)
            preds: List[np.ndarray] = []
            for idx, count in zip(daily_index, daily_count):
                batch = slice(idx, idx + count)
                feat = torch.from_numpy(X_arr[batch]).float().to(self.device)
                c_mat = torch.from_numpy(concept[stock_idx[batch]]).float().to(self.device)
                with torch.no_grad():
                    preds.append(self._model(feat, c_mat).detach().cpu().numpy())
            return np.concatenate(preds)
        else:
            # Fallback: single batch
            feat = torch.from_numpy(X_arr).float().to(self.device)
            # Without instrument index, use default concept row (0)
            c_mat = torch.from_numpy(concept[[0] * len(X_arr)]).float().to(self.device)
            with torch.no_grad():
                return self._model(feat, c_mat).detach().cpu().numpy()

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
