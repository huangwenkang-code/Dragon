"""
PyTorch utility functions adapted from Microsoft Qlib.

Credit:  Microsoft Qlib (MIT License)
  https://github.com/microsoft/qlib
"""

from __future__ import annotations

import os
import torch.nn as nn


def count_parameters(models_or_parameters, unit="m"):
    """Compute the storage size unit of a (or multiple) models.

    Parameters
    ----------
    models_or_parameters : PyTorch model(s) or a list of parameters.
    unit : str
        Storage size unit (``"k"/"kb"``, ``"m"/"mb"``, ``"g"/"gb"``, or None).

    Returns
    -------
    float
        The number of parameters of the given model(s) or parameters.
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2 ** 10
    elif unit in ("mb", "m"):
        counts /= 2 ** 20
    elif unit in ("gb", "g"):
        counts /= 2 ** 30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts


def get_or_create_path(path: str | None) -> str:
    """Return *path* as-is, creating parent directories if needed.

    This replaces qlib's ``get_or_create_path`` utility.
    If *path* is None, a temporary file in the current directory is created.
    """
    if path is None:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".bin", prefix="dragon_model_")
        os.close(fd)
        return path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
