"""
Temporal Routing Adaptor (TRA) model adapted from Microsoft Qlib.

TRA addresses the problem of evolving market regimes by learning multiple
predictors (one per latent trading pattern) and dynamically routing each
sample to the appropriate predictor via a learned router.

The architecture consists of:
1. A backbone model (RNN or Transformer) that encodes temporal features
2. A TRA module with multiple prediction heads and a router that selects
   which head(s) to use based on historical prediction errors

Transport-based optimization (Sinkhorn) aligns predictions with the
oracle assignment to encourage specialization among predictors.

Reference: Heng et al., "Temporal Routing Adaptor for Multi-Regime
Stock Prediction" (2022).

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
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseModel, ArrayLike, SeriesLike
from .pytorch_utils import get_or_create_path

logger = logging.getLogger(__name__)

# Configuration constant
EPS = 1e-12
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Backbone models: RNN and Transformer
# =============================================================================

class RNNet(nn.Module):
    """RNN backbone with optional attention.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Hidden state dimension.
    num_layers : int
        Number of recurrent layers.
    rnn_arch : str
        RNN type: ``"GRU"`` or ``"LSTM"``.
    use_attn : bool
        Whether to use concat attention on top of the RNN.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_arch: str = "GRU",
        use_attn: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_arch = rnn_arch
        self.use_attn = use_attn

        if hidden_size < input_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = None

        self.rnn = getattr(nn, rnn_arch)(
            input_size=min(input_size, hidden_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, F]
        if self.input_proj is not None:
            x = self.input_proj(x)

        rnn_out, last_out = self.rnn(x)
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]
        last_out = last_out.mean(dim=0)

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from PyTorch tutorial)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """Transformer backbone.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Hidden dimension.
    num_layers : int
        Number of encoder layers.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads,
            dropout=dropout,
            d_model=hidden_size,
            dim_feedforward=hidden_size * 4,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, F] -> [T, N, F]
        x = x.permute(1, 0, 2).contiguous()
        x = self.pe(x)
        x = self.input_proj(x)
        out = self.encoder(x)
        return out[-1]  # last timestep


# =============================================================================
# Temporal Routing Adaptor (TRA)
# =============================================================================

class TRAModule(nn.Module):
    """TRA: multiple predictors + learned router.

    Parameters
    ----------
    input_size : int
        Backbone hidden size.
    num_states : int
        Number of latent states / predictors.
    hidden_size : int
        Hidden size of the router.
    rnn_arch : str
        Router RNN type.
    num_layers : int
        Router RNN layers.
    dropout : float
        Dropout.
    tau : float
        Gumbel softmax temperature.
    src_info : str
        Router input: ``"LR"``, ``"TPE"``, or ``"LR_TPE"``.
    """

    def __init__(
        self,
        input_size: int,
        num_states: int = 1,
        hidden_size: int = 8,
        rnn_arch: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.0,
        tau: float = 1.0,
        src_info: str = "LR_TPE",
    ) -> None:
        super().__init__()

        assert src_info in ("LR", "TPE", "LR_TPE"), f"Invalid src_info: {src_info}"

        self.num_states = num_states
        self.tau = tau
        self.rnn_arch = rnn_arch
        self.src_info = src_info

        self.predictors = nn.Linear(input_size, num_states)

        if self.num_states > 1:
            if "TPE" in src_info:
                self.router = getattr(nn, rnn_arch)(
                    input_size=num_states,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.fc = nn.Linear(
                    hidden_size + input_size if "LR" in src_info else hidden_size,
                    num_states,
                )
            else:
                self.fc = nn.Linear(input_size, num_states)

    def reset_parameters(self) -> None:
        for child in self.children():
            child.reset_parameters()

    def forward(
        self, hidden: torch.Tensor, hist_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        preds = self.predictors(hidden)

        if self.num_states == 1:
            return preds, None, None

        if "TPE" in self.src_info:
            out = self.router(hist_loss)[1]  # TPE
            if self.rnn_arch == "LSTM":
                out = out[0]
            out = out.mean(dim=0)
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)  # LR_TPE
        else:
            out = hidden  # LR

        out = self.fc(out)

        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
        prob = torch.softmax(out / self.tau, dim=-1)

        return preds, choice, prob


# =============================================================================
# Transport functions
# =============================================================================

def _shoot_infs(inp_tensor: torch.Tensor) -> torch.Tensor:
    """Replace inf values by the tensor maximum."""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def _sinkhorn(Q: torch.Tensor, n_iters: int = 3, epsilon: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)
        Q = _shoot_infs(Q)
        for _ in range(n_iters):
            Q = Q / Q.sum(dim=0, keepdim=True)
            Q = Q / Q.sum(dim=1, keepdim=True)
    return Q


def _loss_fn(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    mask = (xmin == xmax).squeeze()
    x = (x - xmin) / (xmax - xmin + EPS)
    x[mask] = 1
    return x


def _transport_sample(
    all_preds: torch.Tensor,
    label: torch.Tensor,
    choice: torch.Tensor,
    prob: torch.Tensor,
    hist_loss: torch.Tensor,
    alpha: float,
    transport_method: str,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample-wise optimal transport routing."""
    assert all_preds.shape == choice.shape
    assert len(all_preds) == len(label)

    all_loss = torch.zeros_like(all_preds)
    mask = ~torch.isnan(label)
    all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)

    L = _minmax_norm(all_loss.detach())
    Lh = L * alpha + _minmax_norm(hist_loss) * (1 - alpha)
    Lh = _minmax_norm(Lh)
    P = _sinkhorn(-Lh)

    if transport_method == "router":
        if training:
            pred = (all_preds * choice).sum(dim=1)
        else:
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]
    else:
        pred = (all_preds * P).sum(dim=1)

    if transport_method == "router":
        loss = _loss_fn(pred, label)
    else:
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


# =============================================================================
# Trainer wrapper
# =============================================================================

class TRAModel(BaseModel):
    """Temporal Routing Adaptor training wrapper.

    .. note::
        This is a simplified adaptation. The full Qlib TRA implementation
        uses ``MTSDatasetH`` with memory-based routing and per-day
        transport. This wrapper supports the core mechanisms:
        multi-predictor TRA module + optimal transport routing.

    Parameters
    ----------
    model_type : str
        Backbone model: ``"RNN"`` or ``"Transformer"``.
    model_config : dict
        Keyword arguments for the backbone model.
    tra_config : dict
        Keyword arguments for the TRA module.
    lr : float
        Learning rate.
    n_epochs : int
        Max training epochs.
    early_stop : int
        Patience for early stopping.
    lamb : float
        Regularization weight for the router.
    rho : float
        Exponential decay rate for ``lamb``.
    alpha : float
        Fusion parameter for transport loss matrix.
    transport_method : str
        ``"none"`` (mean of all predictors), ``"router"``, or ``"oracle"``.
    device : str | None
        torch device.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        model_type: str = "RNN",
        model_config: Optional[Dict[str, Any]] = None,
        tra_config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        n_epochs: int = 500,
        early_stop: int = 50,
        lamb: float = 0.0,
        rho: float = 0.99,
        alpha: float = 1.0,
        transport_method: str = "none",
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        assert transport_method in ("none", "router", "oracle"), \
            f"Invalid transport_method: {transport_method}"

        self.model_type = model_type
        self.model_config = model_config or {}
        self.tra_config = tra_config or {}
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.lamb = lamb
        self.rho = rho
        self.alpha = alpha
        self.transport_method = transport_method

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._init_model()

        self._global_step = -1
        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    def _init_model(self) -> None:
        """Instantiate backbone and TRA module."""
        logger.info("Initializing TRAModel...")

        if self.model_type == "RNN":
            self._backbone = RNNet(**self.model_config).to(self.device)
        elif self.model_type == "Transformer":
            self._backbone = TransformerNet(**self.model_config).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        logger.info("Backbone: %s", self._backbone)

        self._tra = TRAModule(
            input_size=self._backbone.output_size, **self.tra_config
        ).to(self.device)
        logger.info("TRA: %s", self._tra)

        self._optimizer = optim.Adam(
            list(self._backbone.parameters()) + list(self._tra.parameters()), lr=self.lr
        )

    @property
    def model(self) -> Tuple[nn.Module, TRAModule]:
        return self._backbone, self._tra

    @property
    def use_gpu(self) -> bool:
        return self.device.type != "cpu"

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        self._backbone.train()
        self._tra.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            self._global_step += 1
            batch_idx = indices[i : i + batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)

            # Reshape for backbone: [B, T, F]
            # TODO: adapt to dragon-engine Dataset interface for proper reshaping
            if feat.dim() == 2:
                # Assume flat input [B, F*T] -> reshape
                d_feat = self.model_config.get("input_size", 20)
                feat = feat.reshape(len(feat), d_feat, -1).permute(0, 2, 1)

            hidden = self._backbone(feat)

            # Create dummy hist_loss for single-state or when no memory
            num_states = self.tra_config.get("num_states", 1)
            hist_loss = torch.zeros(len(hidden), num_states).to(self.device)

            all_preds, choice, prob = self._tra(hidden, hist_loss)

            if self.transport_method != "none" and num_states > 1:
                assert choice is not None and prob is not None
                loss, pred, L, P = _transport_sample(
                    all_preds, lbl, choice, prob,
                    hist_loss, self.alpha,
                    self.transport_method, training=True,
                )
                decay = self.rho ** (self._global_step // 100)
                lamb = self.lamb * decay
                reg = prob.log().mul(P).sum(dim=1).mean()
                loss = loss - lamb * reg
            else:
                pred = all_preds.mean(dim=1)
                loss = F.mse_loss(pred, lbl)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                list(self._backbone.parameters()) + list(self._tra.parameters()), 3.0
            )
            self._optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _test_epoch(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Tuple[float, float]:
        self._backbone.eval()
        self._tra.eval()

        losses: List[float] = []
        indices = np.arange(len(X))

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch_idx = indices[i : i + batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)

            if feat.dim() == 2:
                d_feat = self.model_config.get("input_size", 20)
                feat = feat.reshape(len(feat), d_feat, -1).permute(0, 2, 1)

            num_states = self.tra_config.get("num_states", 1)
            hist_loss = torch.zeros(len(feat), num_states).to(self.device)

            with torch.no_grad():
                hidden = self._backbone(feat)
                all_preds, choice, prob = self._tra(hidden, hist_loss)

                if self.transport_method != "none" and num_states > 1 and prob is not None:
                    pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]
                else:
                    pred = all_preds.mean(dim=1)

                loss = F.mse_loss(pred, lbl)
                losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        # Metric: higher is better -> negated loss
        return mean_loss, -mean_loss

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
        batch_size: int = 1024,
        **kwargs: Any,
    ) -> "TRAModel":
        """Train the TRA model.

        Parameters
        ----------
        X, y : training data (numpy arrays, shape [N, T*F]).
        X_valid, y_valid : validation data.
        save_path : str | None
            Save path for best model state.
        batch_size : int
            Mini-batch size.
        """
        X_train = self._as_array(X)
        y_train = self._as_array(y).ravel()
        X_v = self._as_array(X_valid) if X_valid is not None else X_train
        y_v = self._as_array(y_valid).ravel() if y_valid is not None else y_train

        stop_steps = 0
        best_score = -np.inf
        best_params = {
            "backbone": copy.deepcopy(self._backbone.state_dict()),
            "tra": copy.deepcopy(self._tra.state_dict()),
        }
        self._evals_result = {"train": [], "valid": []}

        logger.info("Training TRA...")
        for epoch in range(self.n_epochs):
            logger.info("Epoch %d:", epoch)
            self._train_epoch(X_train, y_train, batch_size)
            train_loss, train_score = self._test_epoch(X_train, y_train, batch_size)
            val_loss, val_score = self._test_epoch(X_v, y_v, batch_size)
            logger.info("train %.6f, valid %.6f", train_score, val_score)

            self._evals_result["train"].append(train_score)
            self._evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_params = {
                    "backbone": copy.deepcopy(self._backbone.state_dict()),
                    "tra": copy.deepcopy(self._tra.state_dict()),
                }
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._backbone.load_state_dict(best_params["backbone"])
        self._tra.load_state_dict(best_params["tra"])
        if save_path is not None:
            torch.save(best_params, get_or_create_path(save_path))

        if self.use_gpu:
            torch.cuda.empty_cache()
        self._fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return scalar predictions for each sample in *X*."""
        if not self._fitted:
            raise RuntimeError("TRAModel is not fitted yet. Call fit() first.")
        X_arr = self._as_array(X)
        self._backbone.eval()
        self._tra.eval()

        n = len(X_arr)
        # Use a reasonable batch size for inference
        bs = min(2048, n)
        preds: List[np.ndarray] = []
        num_states = self.tra_config.get("num_states", 1)

        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = torch.from_numpy(X_arr[start:end]).float().to(self.device)

            if batch.dim() == 2:
                d_feat = self.model_config.get("input_size", 20)
                batch = batch.reshape(len(batch), d_feat, -1).permute(0, 2, 1)

            hist_loss = torch.zeros(len(batch), num_states).to(self.device)
            with torch.no_grad():
                hidden = self._backbone(batch)
                all_preds, choice, prob = self._tra(hidden, hist_loss)

                if self.transport_method != "none" and num_states > 1 and prob is not None:
                    out = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]
                else:
                    out = all_preds.mean(dim=1)

                preds.append(out.detach().cpu().numpy())

        return np.concatenate(preds)

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
