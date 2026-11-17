"""Read-only inputs shared by every ST sweep."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jax import Array

from krystal.transport.base import Transformation

LogProbFn = Callable[[Array | int, Array], Array]


@dataclass(frozen=True)
class STContext:
    """Everything a sweep reads but never changes.

    Settings (Python values):

    - ``flow`` / ``log_prob``: the transport map and the per-level log-density.
    - ``n_levels``: ladder size ``T``.
    - ``step_size`` / ``n_local``: RW-MH scale multiplier and steps per sweep.

    Arrays (frozen snapshot of the current ``PipelineState``):

    - ``stacked_params``: flow params stacked on a leading edge axis.
    - ``log_lambda``: log ST weights, shape ``(n_levels,)``.
    - ``cov_chols``: ``chol(Σ_level)`` per level, shape ``(n_levels, d, d)``.
    """

    flow: Transformation
    log_prob: LogProbFn
    n_levels: int
    step_size: float
    n_local: int
    stacked_params: Any
    log_lambda: Array
    cov_chols: Array
