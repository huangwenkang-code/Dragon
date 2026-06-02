"""
ADARNN (Adaptive RNN) model adapted from Microsoft Qlib.

ADARNN addresses temporal covariate shift in financial time series by
learning to transfer knowledge across different time periods. It uses:
1. A GRU backbone shared across time periods
2. Temporal distribution characterization via transfer loss (MMD, CORAL, etc.)
3. AdaRNN or Boosting-based temporal weight update

Reference: Du et al., "AdaRNN: Adaptive Learning and Forecasting of
Time Series" (CIKM 2021).

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
# Transfer loss modules (preserved from Qlib source)
# =============================================================================

def cosine_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source, target = source.mean(), target.mean()
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        return torch.sigmoid(x)


def adv_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    input_dim: int = 256,
    hidden_dim: int = 512,
) -> torch.Tensor:
    domain_loss = nn.BCELoss()
    adv_net = Discriminator(input_dim, hidden_dim).to(device)
    domain_src = torch.ones(len(source)).to(device)
    domain_tar = torch.zeros(len(target)).to(device)
    domain_src = domain_src.view(domain_src.shape[0], 1)
    domain_tar = domain_tar.view(domain_tar.shape[0], 1)
    reverse_src = ReverseLayerF.apply(source, 1)
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    pred_tar = adv_net(reverse_tar)
    loss_s = domain_loss(pred_src, domain_src)
    loss_t = domain_loss(pred_tar, domain_tar)
    return loss_s + loss_t


def coral_loss(
    source: torch.Tensor, target: torch.Tensor, device: torch.device
) -> torch.Tensor:
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    tmp_s = torch.ones((1, ns)).to(device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    tmp_t = torch.ones((1, nt)).to(device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    loss = (cs - ct).pow(2).sum()
    return loss / (4 * d * d)


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss."""

    def __init__(
        self,
        kernel_type: str = "linear",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ) -> None:
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    @staticmethod
    def _gaussian_kernel(
        source: torch.Tensor,
        target: torch.Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def _linear_mmd(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        delta = X.mean(axis=0) - Y.mean(axis=0)
        return delta.dot(delta.T)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "linear":
            return self._linear_mmd(source, target)
        elif self.kernel_type == "rbf":
            batch_size = int(source.size(0))
            kernels = self._gaussian_kernel(
                source, target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma,
            )
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss
        raise ValueError(f"Unknown kernel type: {self.kernel_type}")


class Mine(nn.Module):
    """Mutual Information Neural Estimator."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512) -> None:
        super().__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        return self.fc2(h1)


class MineEstimator(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512) -> None:
        super().__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Y_shffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shffle)
        ret = torch.mean(loss_joint) - torch.log(torch.mean(torch.exp(loss_marginal)))
        return -ret


def kl_div(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    criterion = nn.KLDivLoss(reduction="batchmean")
    return criterion(source.log(), target)


def js_div(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    M = 0.5 * (source + target)
    return 0.5 * (kl_div(source, M) + kl_div(target, M))


class TransferLoss:
    """Transfer loss dispatcher.

    Supported loss types: mmd, mmd_rbf, coral, cosine, kl, js, mine, adv.
    """

    def __init__(
        self, loss_type: str = "cosine", input_dim: int = 512, device: Optional[torch.device] = None
    ) -> None:
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.device = device or torch.device("cpu")

    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.loss_type in ("mmd_lin", "mmd"):
            return MMDLoss(kernel_type="linear")(X, Y)
        elif self.loss_type == "coral":
            return coral_loss(X, Y, self.device)
        elif self.loss_type in ("cosine", "cos"):
            return 1 - cosine_loss(X, Y)
        elif self.loss_type == "kl":
            return kl_div(X, Y)
        elif self.loss_type == "js":
            return js_div(X, Y)
        elif self.loss_type == "mine":
            mine_model = MineEstimator(input_dim=self.input_dim, hidden_dim=60).to(self.device)
            return mine_model(X, Y)
        elif self.loss_type == "adv":
            return adv_loss(X, Y, self.device, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == "mmd_rbf":
            return MMDLoss(kernel_type="rbf")(X, Y)
        elif self.loss_type == "pairwise":
            n, d = X.shape
            m = Y.shape[0]
            a = X.unsqueeze(1).expand(n, m, d)
            b = Y.unsqueeze(0).expand(n, m, d)
            pair_mat = torch.pow(a - b, 2).sum(2)
            return torch.norm(pair_mat)
        raise ValueError(f"Unknown loss type: {self.loss_type}")


# =============================================================================
# AdaRNN network module
# =============================================================================

class AdaRNN(nn.Module):
    """Adaptive RNN with temporal distribution characterization.

    Parameters
    ----------
    use_bottleneck : bool
        Whether to use a bottleneck layer.
    bottleneck_width : int
        Bottleneck hidden dimension.
    n_input : int
        Number of input features per timestep.
    n_hiddens : list[int]
        Hidden sizes for each GRU layer.
    n_output : int
        Output dimension.
    dropout : float
        Dropout probability.
    len_seq : int
        Sequence length (number of timesteps).
    model_type : str
        ``"AdaRNN"`` (gate-based) or ``"Boosting"``.
    trans_loss : str
        Transfer loss type for domain adaptation.
    """

    def __init__(
        self,
        use_bottleneck: bool = False,
        bottleneck_width: int = 256,
        n_input: int = 128,
        n_hiddens: Optional[List[int]] = None,
        n_output: int = 6,
        dropout: float = 0.0,
        len_seq: int = 9,
        model_type: str = "AdaRNN",
        trans_loss: str = "mmd",
    ) -> None:
        super().__init__()
        if n_hiddens is None:
            n_hiddens = [64, 64]
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        self.device: torch.device = torch.device("cpu")

        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=dropout,
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == "AdaRNN":
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(len_seq * self.hiddens[i] * 2, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for _ in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self) -> None:
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    # ------------------------------------------------------------------
    # Shared GRU feature extraction
    # ------------------------------------------------------------------

    def gru_features(
        self, x: torch.Tensor, predict: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
        x_input = x
        out_lis: List[torch.Tensor] = []
        out_weight_list: List[torch.Tensor] = [] if self.model_type == "AdaRNN" else []
        out = None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == "AdaRNN" and not predict:
                out_gate = self._process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list if self.model_type == "AdaRNN" else None

    def _process_gate_weight(self, out: torch.Tensor, index: int) -> torch.Tensor:
        x_s = out[0 : int(out.shape[0] // 2)]
        x_t = out[out.shape[0] // 2 : out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        return self.softmax(weight).squeeze()

    @staticmethod
    def _get_features(output_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        fea_list_src = []
        fea_list_tar = []
        for fea in output_list:
            fea_list_src.append(fea[0 : fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2 :])
        return fea_list_src, fea_list_tar

    # ------------------------------------------------------------------
    # Pre-training forward (AdaRNN gate-based)
    # ------------------------------------------------------------------

    def forward_pre_train(
        self, x: torch.Tensor, len_win: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self._get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion_transfer = TransferLoss(
                loss_type=self.trans_loss, input_dim=n.shape[2], device=self.device
            )
            h_start = 0
            for j in range(h_start, self.len_seq, 1):
                i_start = max(j - len_win, 0)
                i_end = min(j + len_win, self.len_seq - 1)
                for k in range(i_start, i_end + 1):
                    weight = (
                        out_weight_list[i][j]
                        if self.model_type == "AdaRNN"
                        else 1 / (self.len_seq - h_start) * (2 * len_win + 1)
                    )
                    loss_transfer = loss_transfer + weight * criterion_transfer.compute(
                        n[:, j, :], out_list_t[i][:, k, :]
                    )
        return fc_out, loss_transfer, out_weight_list

    # ------------------------------------------------------------------
    # Boosting forward
    # ------------------------------------------------------------------

    def forward_Boosting(
        self, x: torch.Tensor, weight_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self._get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        if weight_mat is None:
            weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq)).to(self.device)
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion_transfer = TransferLoss(
                loss_type=self.trans_loss, input_dim=n.shape[2], device=self.device
            )
            for j in range(self.len_seq):
                loss_trans = criterion_transfer.compute(n[:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight

    def update_weight_Boosting(
        self, weight_mat: torch.Tensor, dist_old: torch.Tensor, dist_new: torch.Tensor
    ) -> torch.Tensor:
        epsilon = 1e-5
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out


# =============================================================================
# Helper
# =============================================================================

def _get_index(num_domain: int = 2) -> List[Tuple[int, int]]:
    """Generate pairwise domain index pairs."""
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index


# =============================================================================
# Trainer wrapper
# =============================================================================

class ADARNNModel(BaseModel):
    """ADARNN training wrapper with temporal domain adaptation.

    Key differences from standard RNN training:
    - The training data is split into ``n_splits`` temporal chunks
      (domains) that serve as source/target pairs for transfer learning.
    - A pre-training phase runs for ``pre_epoch`` epochs to warm up the
      gating mechanism.
    - A boosting phase follows with adaptive temporal weighting.

    Parameters
    ----------
    d_feat : int
        Features per timestep.
    hidden_size : int
        GRU hidden size.
    num_layers : int
        Number of GRU layers.
    dropout : float
        Dropout.
    n_epochs : int
        Max training epochs.
    pre_epoch : int
        Number of pre-training (gate warm-up) epochs.
    dw : float
        Weight of the transfer loss term.
    loss_type : str
        Transfer loss type (``"mmd"``, ``"cosine"``, ``"coral"``, etc.).
    len_seq : int
        Sequence length (number of timesteps).
    len_win : int
        Window size for pre-training temporal matching.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    early_stop : int
        Patience for early stopping.
    n_splits : int
        Number of temporal domain splits.
    loss : str
        Supervised loss name (``"mse"``).
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
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 200,
        pre_epoch: int = 40,
        dw: float = 0.5,
        loss_type: str = "cosine",
        len_seq: int = 60,
        len_win: int = 0,
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop: int = 20,
        n_splits: int = 2,
        loss: str = "mse",
        optimizer: str = "adam",
        device: Optional[Union[str, int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.pre_epoch = pre_epoch
        self.dw = dw
        self.loss_type = loss_type
        self.len_seq = len_seq
        self.len_win = len_win
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.n_splits = n_splits
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

        n_hiddens = [hidden_size for _ in range(num_layers)]
        self._model = AdaRNN(
            use_bottleneck=False,
            bottleneck_width=64,
            n_input=d_feat,
            n_hiddens=n_hiddens,
            n_output=1,
            dropout=dropout,
            model_type="AdaRNN",
            len_seq=len_seq,
            trans_loss=loss_type,
        )
        self._model.device = self.device

        logger.info("ADARNN model size: %.4f MB", count_parameters(self._model))

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
    def model(self) -> AdaRNN:
        return self._model

    @property
    def use_gpu(self) -> bool:
        return self.device.type != "cpu"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transform_type(self, init_weight: List[torch.Tensor]) -> torch.Tensor:
        weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_ada_rnn(
        self,
        train_loader_list: List[Any],
        epoch: int,
        dist_old: Optional[torch.Tensor] = None,
        weight_mat: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._model.train()
        criterion = nn.MSELoss()
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        out_weight_list: Any = None
        for data_all in zip(*train_loader_list):
            self._optimizer.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                feature, label_reg = data[0].to(self.device).float(), data[1].to(self.device).float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = _get_index(len(data_all) - 1)
            for temp_index in index:
                s1, s2 = temp_index[0], temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).to(self.device)
            for i, n in enumerate(index):
                feature_s = list_feat[n[0]]
                feature_t = list_feat[n[1]]
                label_reg_s = list_label[n[0]]
                label_reg_t = list_label[n[1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                if epoch < self.pre_epoch:
                    pred_all, loss_transfer, out_weight_list = self._model.forward_pre_train(
                        feature_all, len_win=self.len_win
                    )
                else:
                    pred_all, loss_transfer, dist, weight_mat = self._model.forward_Boosting(
                        feature_all, weight_mat
                    )
                    dist_mat = dist_mat + dist
                pred_s = pred_all[0 : feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0) :]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)
                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer

            self._optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 3.0)
            self._optimizer.step()

        if epoch >= self.pre_epoch:
            if epoch > self.pre_epoch:
                weight_mat = self._model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
            return weight_mat, dist_mat
        else:
            weight_mat = self._transform_type(out_weight_list)
            return weight_mat, None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_metrics(pred: pd.DataFrame) -> Dict[str, float]:
        """Compute IC, Rank IC, ICIR, RICIR, MSE for a pred DataFrame."""
        res: Dict[str, float] = {}
        ic = pred.groupby(level=0, group_keys=False).apply(lambda x: x["label"].corr(x["score"]))
        rank_ic = pred.groupby(level=0, group_keys=False).apply(
            lambda x: x["label"].corr(x["score"], method="spearman")
        )
        res["ic"] = float(ic.mean())
        res["icir"] = float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0
        res["ric"] = float(rank_ic.mean())
        res["ricir"] = float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() > 0 else 0.0
        res["mse"] = float(-(pred["label"] - pred["score"]).mean())
        res["loss"] = res["mse"]
        return res

    def _infer_array(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        n = len(X)
        x_vals = X.reshape(n, self.d_feat, -1).transpose(0, 2, 1)
        preds: List[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = torch.from_numpy(x_vals[start:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self._model.predict(batch).detach().cpu().numpy())

        return np.concatenate(preds)

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
    ) -> "ADARNNModel":
        """Train ADARNN with temporal domain adaptation.

        *X* should be a DataFrame with a ``(datetime, instrument)``
        MultiIndex so the data can be split into temporal domains.

        Parameters
        ----------
        X, y : training data.
        X_valid, y_valid : validation data.
        save_path : str | None
            If provided, the best model state is saved to this path.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ADARNNModel.fit() requires X as a pd.DataFrame with MultiIndex")
        if X_valid is not None and not isinstance(X_valid, pd.DataFrame):
            raise TypeError("ADARNNModel.fit() requires X_valid as a pd.DataFrame")

        # Split training data by time into n_splits domains
        # TODO: adapt to dragon-engine Dataset interface for temporal domain splitting
        days = X.index.get_level_values(level=0).unique()
        train_splits = np.array_split(days, self.n_splits)
        train_data_splits = []
        for s in train_splits:
            subset = X.loc[s[0]:s[-1]]
            train_data_splits.append(subset)

        X_train = self._as_array(X)
        y_train = self._as_array(y).ravel()
        X_v = self._as_array(X_valid) if X_valid is not None else X_train
        y_v = self._as_array(y_valid).ravel() if y_valid is not None else y_train

        # Build DataLoaders for each domain split
        # TODO: adapt to dragon-engine Dataset interface for domain loaders
        from torch.utils.data import DataLoader, TensorDataset

        def _make_loader(split_df: pd.DataFrame) -> DataLoader:
            idx = X.index.get_indexer(split_df.index)
            feat = torch.tensor(
                X_train[idx].reshape(-1, self.d_feat, self.len_seq).transpose(0, 2, 1),
                dtype=torch.float32,
            )
            lbl = torch.tensor(y_train[idx].reshape(-1), dtype=torch.float32)
            ds = TensorDataset(feat, lbl)
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        train_loader_list = [_make_loader(s) for s in train_data_splits]

        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self._model.state_dict())
        self._evals_result = {"train": [], "valid": []}
        weight_mat: Optional[torch.Tensor] = None
        dist_mat: Optional[torch.Tensor] = None

        logger.info("Training ADARNN...")
        for epoch in range(self.n_epochs):
            logger.info("Epoch %d:", epoch)
            weight_mat, dist_mat = self._train_ada_rnn(
                train_loader_list, epoch, dist_mat, weight_mat
            )

            # Evaluate
            train_preds = self._infer_array(X_train)
            val_preds = self._infer_array(X_v)
            train_score = -np.mean((train_preds - y_train) ** 2)
            val_score = -np.mean((val_preds - y_v) ** 2)
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
            raise RuntimeError("ADARNNModel is not fitted yet. Call fit() first.")
        X_arr = self._as_array(X)
        return self._infer_array(X_arr)

    @property
    def evals_result(self) -> Dict[str, List[float]]:
        """Training history (score per epoch)."""
        return self._evals_result
