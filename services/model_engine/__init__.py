"""dragon-engine model engine — adapted from Microsoft Qlib model zoo.

Available architectures (self-contained, inheriting from BaseModel):
  - XGBoost        — tabular factor scoring (fast, calibrated)
  - GATs           — cross-stock graph attention for leader detection
  - Transformer    — temporal sequence modeling with positional encoding
  - TCN            — temporal convolutional network (dilated causal convolutions)
  - LSTM           — long short-term memory baseline
  - GRU            — gated recurrent unit (faster LSTM alternative)
  - ALSTM          — attentive LSTM (attention over hidden states)
  - SFM            — state frequency memory (frequency-domain modeling)
  - TRA            — temporal routing adaptor (multi-timescale routing)
  - TabNet         — attentive tabular learning
  - HIST           — historical information-based stock trend
  - Localformer    — local attention for financial time series
  - ADARNN         — adaptive RNN (handles concept drift)

Ensemble:
  - ModelEnsemble  — weighted fusion of all model predictions
"""

from .base import BaseModel
from .ensemble import ModelEnsemble

__all__ = [
    "BaseModel",
    "ModelEnsemble",
]
