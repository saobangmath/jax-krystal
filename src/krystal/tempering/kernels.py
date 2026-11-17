"""Pure-JAX sampling kernels for lifted Simulated Tempering.

One ST *sweep* moves every particle twice:

1. :func:`rw_metropolis` — random-walk MH inside the particle's current level.
2. :func:`ladder_jump` — one lifted MH jump ``I → I + V`` using the flow.

Both kernels act on a **single** particle. :func:`st_sweep` ``vmap``s them over
the whole ensemble. Everything that stays fixed during a run (flow, densities,
weights, covariances) is bundled in :class:`~krystal.tempering.context.STContext`,
so each kernel only takes the key plus the particle's own
``(x, level, direction)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax, random
from jax.random import Key

from krystal.tempering.constants import RW_MH_OPTIMAL_SCALE
from krystal.tempering.context import STContext


def rw_metropolis(key: Key, x: Array, level: Array, ctx: STContext) -> Array:
    """Run ``ctx.n_local`` random-walk MH steps targeting ``π_level``.

    Proposal: ``x' = x + step_size · (2.38 / √d) · chol @ ε`` with
    ``ε ~ N(0, I)`` and ``Σ_level = chol cholᵀ``. For ``step_size = 1`` this is
    ``N(x, (2.38² / d) Σ_level)``. The proposal is symmetric, so the MH ratio is
    just ``π_level(x') / π_level(x)``.
    """
    dim = x.shape[-1]
    scale = ctx.step_size * RW_MH_OPTIMAL_SCALE / jnp.sqrt(dim)
    chol = ctx.cov_chols[level]

    def one_step(x: Array, step_key: Key) -> tuple[Array, None]:
        noise_key, accept_key = random.split(step_key)
        proposal = x + scale * (chol @ random.normal(noise_key, x.shape))
        log_alpha = (
            ctx.log_prob(level, proposal[None, :])[0]
            - ctx.log_prob(level, x[None, :])[0]
        )
        accept = jnp.log(random.uniform(accept_key)) < log_alpha
        return jnp.where(accept, proposal, x), None

    x, _ = lax.scan(one_step, x, random.split(key, ctx.n_local))
    return x


def ladder_jump(
    key: Key,
    x: Array,
    level: Array,
    direction: Array,
    ctx: STContext,
) -> tuple[Array, Array, Array]:
    """Lifted MH jump from ``level`` to ``level + direction``.

    The flow is the proposal: going up uses ``forward`` of edge ``level``,
    going down uses ``inverse`` of edge ``level - 1``. Accept with probability
    ``min(1, exp(log_alpha))``, where::

        log_alpha = log λ_t + log π_t(y) - log λ_i - log π_i(x) + log|det J|

    On accept the particle moves and keeps its direction. On reject, or when
    the target level is off the ladder, it stays put and ``direction`` flips.

    Returns ``(x, level, direction)``.
    """
    target = level + direction
    off_ladder = (target < 0) | (target > ctx.n_levels - 1)
    target = jnp.clip(target, 0, ctx.n_levels - 1)  # keep indexing valid when off_ladder

    going_up = direction > 0
    edge = jnp.where(going_up, level, target)  # bridge between level and target
    params = jax.tree.map(lambda p: p[edge], ctx.stacked_params)

    # ``going_up`` is traced under ``vmap``, so evaluate both maps and select.
    y_up, log_det_up = ctx.flow.forward(params, x[None, :])
    y_down, log_det_down = ctx.flow.inverse(params, x[None, :])
    y = jnp.where(going_up, y_up[0], y_down[0])
    log_det = jnp.where(going_up, log_det_up[0], log_det_down[0])

    log_alpha = (
        ctx.log_lambda[target]
        + ctx.log_prob(target, y[None, :])[0]
        - ctx.log_lambda[level]
        - ctx.log_prob(level, x[None, :])[0]
        + log_det
    )
    accept = ~off_ladder & (jnp.log(random.uniform(key)) < log_alpha)

    new_x = jnp.where(accept, y, x)
    new_level = jnp.where(accept, target, level)
    new_direction = jnp.where(accept, direction, -direction).astype(direction.dtype)
    return new_x, new_level, new_direction


def st_sweep(
    key: Key,
    x: Array,
    levels: Array,
    directions: Array,
    ctx: STContext,
) -> tuple[Key, Array, Array, Array, Array]:
    """One ST sweep over all particles (``vmap`` of RW then ladder jump).

    Returns ``(key, x, levels, directions, visits)`` where ``visits`` is the
    number of particles on each level after the sweep, shape ``(n_levels,)``.
    """
    key, sweep_key = random.split(key)
    particle_keys = random.split(sweep_key, x.shape[0])

    def one_particle(
        particle_key: Key, x: Array, level: Array, direction: Array
    ) -> tuple[Array, Array, Array]:
        rw_key, jump_key = random.split(particle_key)
        x = rw_metropolis(rw_key, x, level, ctx)
        return ladder_jump(jump_key, x, level, direction, ctx)

    x, levels, directions = jax.vmap(one_particle)(
        particle_keys, x, levels, directions
    )
    visits = jnp.bincount(levels, length=ctx.n_levels)
    return key, x, levels, directions, visits
