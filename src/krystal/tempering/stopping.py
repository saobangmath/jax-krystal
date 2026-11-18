"""Early stopping for the outer CRAFT-ST loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from krystal.tempering.constants import (
    DEFAULT_PLATEAU_SMOOTHING,
    DEFAULT_PLATEAU_TOL,
)


@dataclass
class PlateauStopper:
    """Stop when the smoothed flow loss has not improved for ``patience`` iterations.

    Call :meth:`update` once per outer iteration with the per-edge
    ``state.flow_losses``. The losses are averaged over edges and smoothed with
    an exponential moving average, since each one is a noisy batch estimate.
    An iteration counts as an improvement only if the smoothed loss drops below
    the best so far by more than ``tol * max(1, |best|)``.

    Nothing is tracked until every edge has a finite loss (edges get one only
    after particles reach their source level), so early warm-up never triggers
    a stop. The absolute loss is offset by unknown log-normalizers, which is why
    only the trend is used.
    """

    patience: int
    tol: float = DEFAULT_PLATEAU_TOL
    smoothing: float = DEFAULT_PLATEAU_SMOOTHING
    best: float | None = field(default=None, init=False)
    ema: float | None = field(default=None, init=False)
    bad_iters: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

    def update(self, flow_losses: Array) -> bool:
        """Record this iteration's losses; return ``True`` when training should stop."""
        if not bool(jnp.all(jnp.isfinite(flow_losses))):
            return False
        value = float(jnp.mean(flow_losses))
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.smoothing * self.ema + (1.0 - self.smoothing) * value

        if self.best is None or self.ema < self.best - self.tol * max(1.0, abs(self.best)):
            self.best = self.ema
            self.bad_iters = 0
        else:
            self.bad_iters += 1
        return self.bad_iters >= self.patience
