"""Simple annealing schedules.

These are placeholders to keep the package useful while OT cores land.
Conventions: schedules return temperature (or mixing) values over ``[0, 1]``
time, with higher values typically meaning a hotter / easier distribution —
exact semantics will be documented per algorithm as they are added.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def linear(n_steps: int, *, start: float = 1.0, end: float = 0.0) -> Array:
    """Linear schedule from ``start`` to ``end`` inclusive."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return jnp.linspace(start, end, n_steps)


def geometric(n_steps: int, *, start: float = 1.0, end: float = 1e-3) -> Array:
    """Geometric schedule from ``start`` to ``end`` inclusive."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be positive for a geometric schedule")
    return jnp.geomspace(start, end, n_steps)
