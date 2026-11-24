"""Pipeline state: ensemble + flow / weight parameters for Simulated Tempering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.random import Key
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class LocalCovState:
    """Per-level Welford stats and the derived RW proposal covariance.

    Fixed-size summaries only — no particle history. ``cov`` is
    ``M2 / (n - 1) + ridge`` when ``n >= 2``, otherwise ``I``.
    """

    n: Array
    """Sample count per level, shape ``(n_levels,)``."""

    mean: Array
    """Running mean per level, shape ``(n_levels, dim)``."""

    m2: Array
    """Sum of squared deviations ``M2`` per level, shape ``(n_levels, dim, dim)``."""

    cov: Array
    """Proposal covariance ``Σ`` per level, shape ``(n_levels, dim, dim)``."""

    @classmethod
    def init(cls, n_levels: int, dim: int, dtype=jnp.float32) -> LocalCovState:
        """Cold start: empty Welford stats and isotropic ``Σ = I``."""
        eye = jnp.eye(dim, dtype=dtype)
        return cls(
            n=jnp.zeros(n_levels, dtype=dtype),
            mean=jnp.zeros((n_levels, dim), dtype=dtype),
            m2=jnp.zeros((n_levels, dim, dim), dtype=dtype),
            cov=jnp.broadcast_to(eye, (n_levels, dim, dim)).copy(),
        )

    def tree_flatten(self):
        return (self.n, self.mean, self.m2, self.cov), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@register_pytree_node_class
@dataclass
class PipelineState:
    """Ensemble + flow / weight parameters for one Pipeline."""

    x: Array
    """Particle configurations, shape ``(n_particles, dim)``."""

    levels: Array
    """Integer ladder index per particle, shape ``(n_particles,)``."""

    directions: Array
    """Persistent direction ``V ∈ {+1, -1}`` per particle, shape ``(n_particles,)``."""

    flow_params: list[Any]
    """Parameters for each edge ``i → i+1`` (length ``n_levels - 1``)."""

    log_lambda: Array
    """Log tempering weights, shape ``(n_levels,)``."""

    key: Key
    """JAX PRNG key (``jax.random.Key`` / ``jax.random.key``)."""

    opt_states: list[Any]
    """Optax state per edge."""

    counts: Array
    """Visit counts from the last ``implement`` call, shape ``(n_levels,)``."""

    local_cov: LocalCovState
    """Welford adaptive proposal covariance (stats + derived ``Σ``)."""

    def tree_flatten(self):
        children = (
            self.x,
            self.levels,
            self.directions,
            self.flow_params,
            self.log_lambda,
            self.key,
            self.opt_states,
            self.counts,
            self.local_cov,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)
