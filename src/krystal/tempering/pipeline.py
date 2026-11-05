"""CRAFT-style tempering pipeline: learn flow → implement → tune λ.

Outer loop (scaffold only — stage bodies land later):

1. **Learn flow** — fit maps ``φ_{i→i+1}`` between neighboring levels.
2. **Implement** — propose ``y = φ(x)`` and accept/reject level moves.
3. **Tune weight** — update ladder weights ``λ``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Pipeline:
    """Iterative ST + transport pipeline over a tempered ladder.

    Parameters
    ----------
    n_levels:
        Number of tempered levels ``0, ..., T`` (``T = n_levels - 1``).
    """

    n_levels: int

    def learn_flow(self, state: Any) -> Any:
        """Train transport maps between consecutive tempered levels."""
        raise NotImplementedError

    def implement(self, state: Any) -> Any:
        """Apply learned maps to propose and accept/reject transitions."""
        raise NotImplementedError

    def tune_weight(self, state: Any) -> Any:
        """Update tempering weights ``λ`` from the current run."""
        raise NotImplementedError

    def step(self, state: Any) -> Any:
        """One outer iteration: learn flow → implement → tune weight."""
        state = self.learn_flow(state)
        state = self.implement(state)
        state = self.tune_weight(state)
        return state

    def run(self, state: Any, *, n_iterations: int) -> Any:
        """Repeat ``step`` for ``n_iterations``."""
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")
        for _ in range(n_iterations):
            state = self.step(state)
        return state
