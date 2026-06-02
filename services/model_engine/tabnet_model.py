"""
TabNet model adapted from Microsoft Qlib.

TabNet uses sequential attention to select which features to reason
from at each decision step. It supports self-supervised pre-training
via a decoder that reconstructs masked features.

Reference: Arik & Pfister, "TabNet: Attentive Interpretable Tabular
Learning" (AAAI 2021).

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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

from .base import BaseModel, ArrayLike, SeriesLike
from .pytorch_utils import count_parameters, get_or_create_path

logger = logging.getLogger(__name__)


# =============================================================================
# TabNet building blocks (preserved from Qlib source)
# =============================================================================


def make_ix_like(input: torch.Tensor, dim: int = 0) -> torch.Tensor:
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """SparseMax activation function."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(
        input: torch.Tensor, dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau = tau / support_size.to(input.dtype)
        return tau, support_size


class GBN(nn.Module):
    """Ghost Batch Normalization."""

    def __init__(self, inp: int, vbs: int = 1024, momentum: float = 0.01) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= self.vbs:
            return self.bn(x)
        chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    """Gated Linear Unit block."""

    def __init__(
        self, inp_dim: int, out_dim: int, fc: Optional[nn.Module] = None, vbs: int = 1024
    ) -> None:
        super().__init__()
        self.fc = fc if fc is not None else nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))


class AttentionTransformer(nn.Module):
    """Attention transformer with sparsemax masking."""

    def __init__(self, d_a: int, inp_dim: int, relax: float, vbs: int = 1024) -> None:
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    def forward(self, a: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)
        return mask


class FeatureTransformer(nn.Module):
    """Feature transformer with shared and independent GLU blocks."""

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        shared: Optional[nn.ModuleList],
        n_ind: int,
        vbs: int,
    ) -> None:
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared is not None:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = nn.ModuleList()  # empty
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for _ in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = float(np.sqrt(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.shared) > 0:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    """One decision step of TabNet."""

    def __init__(
        self,
        inp_dim: int,
        n_d: int,
        n_a: int,
        shared: Optional[nn.ModuleList],
        n_ind: int,
        relax: float,
        vbs: int,
    ) -> None:
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)

    def forward(
        self, x: torch.Tensor, a: torch.Tensor, priors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss


class TabNet(nn.Module):
    """TabNet encoder.

    Parameters
    ----------
    inp_dim : int
        Number of input features.
    out_dim : int
        Output feature dimension.
    n_d : int
        Dimension of the decision prediction layer.
    n_a : int
        Dimension of the attention mask.
    n_shared : int
        Number of shared GLU blocks in the feature transformer.
    n_ind : int
        Number of independent GLU blocks.
    n_steps : int
        Number of decision steps.
    relax : float
        Relaxation factor for attention sparsemax (>1 means reuse).
    vbs : int
        Virtual batch size for Ghost BN.
    """

    def __init__(
        self,
        inp_dim: int = 6,
        out_dim: int = 6,
        n_d: int = 64,
        n_a: int = 64,
        n_shared: int = 2,
        n_ind: int = 2,
        n_steps: int = 5,
        relax: float = 1.2,
        vbs: int = 1024,
    ) -> None:
        super().__init__()

        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for _ in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        else:
            self.shared = nn.ModuleList()

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared if len(self.shared) > 0 else None, n_ind, vbs)
        self.steps = nn.ModuleList()
        for _ in range(n_steps - 1):
            self.steps.append(
                DecisionStep(inp_dim, n_d, n_a, self.shared if len(self.shared) > 0 else None, n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d

    def forward(
        self, x: torch.Tensor, priors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss_list: List[torch.Tensor] = []
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, loss = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])
            x_a = x_te[:, self.n_d :]
            sparse_loss_list.append(loss)
        return self.fc(out), sum(sparse_loss_list)


class DecoderStep(nn.Module):
    """One decoder step for TabNet pre-training."""

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        shared: Optional[nn.ModuleList],
        n_ind: int,
        vbs: int,
    ) -> None:
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs)
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fea_tran(x)
        return self.fc(x)


class TabNetDecoder(nn.Module):
    """TabNet decoder used in self-supervised pre-training."""

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        n_shared: int,
        n_ind: int,
        vbs: int,
        n_steps: int,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * out_dim))
            for _ in range(n_shared - 1):
                self.shared.append(nn.Linear(out_dim, 2 * out_dim))
        else:
            self.shared = nn.ModuleList()
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        for _ in range(n_steps):
            self.steps.append(
                DecoderStep(inp_dim, out_dim, self.shared if len(self.shared) > 0 else None, n_ind, vbs)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        for step in self.steps:
            out += step(x)
        return out


class FinetuneNet(nn.Module):
    """Wrapper that adds a final linear layer to the TabNet encoder."""

    def __init__(self, input_dim: int, output_dim: int, trained_model: nn.Module) -> None:
        super().__init__()
        self.model = trained_model
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        return self.fc(self.model(x, priors)[0]).squeeze()


# =============================================================================
# Trainer wrapper
# =============================================================================

class TabNetModel(BaseModel):
    """TabNet training wrapper with optional self-supervised pre-training.

    Parameters
    ----------
    d_feat : int
        Number of input features.
    out_dim : int
        Encoder output dimension.
    final_out_dim : int
        Final prediction dimension.
    batch_size : int
        Training batch size.
    n_d : int
        Decision step output dimension.
    n_a : int
        Attention dimension.
    n_shared : int
        Number of shared GLU blocks.
    n_ind : int
        Number of independent GLU blocks.
    n_steps : int
        Number of decision steps.
    relax : float
        Relaxation factor for attention.
    vbs : int
        Virtual batch size for Ghost BN.
    n_epochs : int
        Max training epochs.
    lr : float
        Learning rate.
    pretrain : bool
        Whether to perform self-supervised pre-training.
    ps : float
        Bernoulli mask probability for pre-training.
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
        d_feat: int = 158,
        out_dim: int = 64,
        final_out_dim: int = 1,
        batch_size: int = 4096,
        n_d: int = 64,
        n_a: int = 64,
        n_shared: int = 2,
        n_ind: int = 2,
        n_steps: int = 5,
        relax: float = 1.3,
        vbs: int = 2048,
        n_epochs: int = 100,
        pretrain_n_epochs: int = 50,
        lr: float = 0.01,
        pretrain: bool = True,
        ps: float = 0.3,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        device: Optional[Union[str, int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.out_dim = out_dim
        self.final_out_dim = final_out_dim
        self.batch_size = batch_size
        self.lr = lr
        self.ps = ps
        self.n_epochs = n_epochs
        self.pretrain_n_epochs = pretrain_n_epochs
        self.loss_name = loss
        self.early_stop = early_stop
        self.pretrain = pretrain
        self.vbs = vbs
        self.relax = relax
        self.n_shared = n_shared
        self.n_ind = n_ind
        self.n_steps = n_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._model: nn.Module = TabNet(
            inp_dim=self.d_feat,
            out_dim=self.out_dim,
            n_d=n_d,
            n_a=n_a,
            n_shared=n_shared,
            n_ind=n_ind,
            n_steps=n_steps,
            relax=relax,
            vbs=vbs,
        ).to(self.device)

        self._decoder = TabNetDecoder(
            self.out_dim, self.d_feat, n_shared, n_ind, vbs, n_steps
        ).to(self.device)

        logger.info("TabNet model size: %.4f MB", count_parameters([self._model, self._decoder]))

        opt = optimizer.lower()
        if opt == "adam":
            self._pretrain_optimizer = optim.Adam(
                list(self._model.parameters()) + list(self._decoder.parameters()), lr=self.lr
            )
            self._train_optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        elif opt in ("gd", "sgd"):
            self._pretrain_optimizer = optim.SGD(
                list(self._model.parameters()) + list(self._decoder.parameters()), lr=self.lr
            )
            self._train_optimizer = optim.SGD(self._model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

        self._evals_result: Dict[str, List[float]] = {"train": [], "valid": []}

    @property
    def model(self) -> nn.Module:
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

    def _pretrain_loss_fn(
        self, f_hat: torch.Tensor, f: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Pre-training reconstruction loss (TabNet paper Sec 4.2)."""
        down_mean = torch.mean(f, dim=0)
        down = torch.sqrt(torch.sum(torch.square(f - down_mean), dim=0))
        up = (f_hat - f) * S
        return torch.sum(torch.square(up / down))

    # ------------------------------------------------------------------
    # Pre-training
    # ------------------------------------------------------------------

    def _pretrain_epoch(self, X_arr: np.ndarray) -> None:
        self._model.train()
        self._decoder.train()
        indices = np.arange(len(X_arr))
        np.random.shuffle(indices)

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            S_mask = torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_in = torch.from_numpy(X_arr[batch_idx]).float() * (1 - S_mask)
            x_out = torch.from_numpy(X_arr[batch_idx]).float() * S_mask

            S_mask = S_mask.to(self.device)
            feat = x_in.float().to(self.device)
            label = x_out.float().to(self.device)
            priors = 1 - S_mask
            vec, sparse_loss = self._model(feat, priors)
            f = self._decoder(vec)
            loss = self._pretrain_loss_fn(label, f, S_mask)

            self._pretrain_optimizer.zero_grad()
            loss.backward()
            self._pretrain_optimizer.step()

    def _pretrain_test_epoch(self, X_arr: np.ndarray) -> float:
        self._model.eval()
        self._decoder.eval()
        indices = np.arange(len(X_arr))
        losses: List[float] = []

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            S_mask = torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_in = torch.from_numpy(X_arr[batch_idx]).float() * (1 - S_mask)
            x_out = torch.from_numpy(X_arr[batch_idx]).float() * S_mask

            feat = x_in.float().to(self.device)
            label = x_out.float().to(self.device)
            S_mask = S_mask.to(self.device)
            priors = 1 - S_mask
            with torch.no_grad():
                vec, sparse_loss = self._model(feat, priors)
                f = self._decoder(vec)
                loss = self._pretrain_loss_fn(label, f, S_mask)
            losses.append(loss.item())

        return float(np.mean(losses))

    # ------------------------------------------------------------------
    # Epoch helpers (train)
    # ------------------------------------------------------------------

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)
            priors = torch.ones(self.batch_size, self.d_feat).to(self.device)

            pred = self._model(feat, priors)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.squeeze()
            loss = self._loss_fn(pred, lbl)

            self._train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self._train_optimizer.step()

    def _test_epoch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self._model.eval()
        losses: List[float] = []
        scores: List[float] = []

        indices = np.arange(len(X))
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            feat = torch.from_numpy(X[batch_idx]).float().to(self.device)
            lbl = torch.from_numpy(y[batch_idx]).float().to(self.device)
            priors = torch.ones(len(batch_idx), self.d_feat).to(self.device)

            with torch.no_grad():
                pred = self._model(feat, priors)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.squeeze()
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
    ) -> "TabNetModel":
        """Train the TabNet model with optional pre-training.

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

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        X_v = np.nan_to_num(X_v, nan=0.0)
        y_v = np.nan_to_num(y_v, nan=0.0)

        # Self-supervised pre-training
        # TODO: adapt to dragon-engine Dataset interface for pre-training data splits
        if self.pretrain:
            logger.info("Pre-training TabNet...")
            best_loss = np.inf
            stop_steps = 0
            for epoch in range(self.pretrain_n_epochs):
                self._pretrain_epoch(X_train)
                train_loss = self._pretrain_test_epoch(X_train)
                valid_loss = self._pretrain_test_epoch(X_v)
                logger.info("pretrain epoch %d: train %.6f, valid %.6f", epoch, train_loss, valid_loss)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    stop_steps = 0
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        logger.info("Pre-training early stop at epoch %d", epoch)
                        break

        # Add fine-tune layer
        self._model = FinetuneNet(self.out_dim, self.final_out_dim, self._model).to(self.device)

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}

        logger.info("Training TabNet...")
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
            raise RuntimeError("TabNetModel is not fitted yet. Call fit() first.")
        X_arr = self._as_array(X)
        self._model.eval()
        preds: List[np.ndarray] = []
        n = len(X_arr)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.from_numpy(X_arr[start:end]).float().to(self.device)
            priors = torch.ones(end - start, self.d_feat).to(self.device)
            with torch.no_grad():
                out = self._model(batch, priors)
                preds.append(out.detach().cpu().numpy())

        return np.concatenate(preds)

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
