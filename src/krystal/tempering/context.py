"""Pipeline config and per-``implement`` snapshots for ST kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jax import Array

from krystal.transport.base import Transformation

LogProbFn = Callable[[Array | int, Array], Array]


@dataclass(frozen=True)
class STContext:
    """Unchanging pipeline settings shared by every ST sweep.

    These never change during training; they are fixed when the
    :class:`~krystal.tempering.pipeline.Pipeline` is constructed.

    - ``flow`` / ``log_prob``: transport map and per-level log-density.
    - ``n_levels``: ladder size ``T``.
    - ``step_size`` / ``n_local``: RW-MH scale multiplier and steps per sweep.

    Arrays that *do* change between outer iterations (``log_lambda``, flow
    params, ``local_cov``) live on :class:`~krystal.tempering.state.PipelineState`.
    For one ``implement`` call they are snapshotted into :class:`SweepInputs`.
    """

    flow: Transformation
    log_prob: LogProbFn
    n_levels: int
    step_size: float
    n_local: int


@dataclass(frozen=True)
class SweepInputs:
    """Frozen array snapshot for one ``implement`` / ST run.

    Built from the current :class:`~krystal.tempering.state.PipelineState`:

    - ``stacked_params``: flow params stacked on a leading edge axis.
    - ``log_lambda``: log ST weights, shape ``(n_levels,)``.
    - ``cov_chols``: ``chol(local_cov)`` per level — derived from state, not
      stored separately. Shape ``(n_levels, d, d)``.
    """

    stacked_params: Any
    log_lambda: Array
    cov_chols: Array
