"""Pipeline state: ensemble + flow / weight parameters for Simulated Tempering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import Array
from jax.random import Key
from jax.tree_util import register_pytree_node_class


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

    local_cov: Array
    """Per-level RW proposal covariance ``Σ``, shape ``(n_levels, dim, dim)``.

    Gelman proposal is ``N(x, (2.38² / d) Σ)``. Starts as ``I``; refresh with
    :meth:`Pipeline.update_local_cov` after particles have mixed.
    """

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
