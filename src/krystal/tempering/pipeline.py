"""CRAFT-style tempering pipeline for Simulated Tempering.

Outer loop order:

1. **Implement** — run ST with current flows; collect level occupancy.
2. **Tune weight** — update ladder weights ``λ`` from occupancy.
3. **Learn flow** — fit ``φ_{i→i+1}`` on the current particle snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import optax
from jax import Array, grad, random
from jax.random import Key
from tqdm.auto import tqdm

from krystal.tempering.constants import (
    DEFAULT_ETA,
    DEFAULT_INNER_SWEEPS,
    DEFAULT_LEARN_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOCAL_STEP_SIZE,
    DEFAULT_N_LOCAL,
    DEFAULT_OUTER_ITERATIONS,
    OCCUPANCY_PSEUDOCOUNT,
    RW_MH_OPTIMAL_SCALE,
)
from krystal.transport.base import Transformation

LogProbFn = Callable[[int, Array], Array]


@dataclass
class PipelineState:
    """Ensemble + flow / weight parameters for one Pipeline."""

    x: Array
    """Particle configurations, shape ``(n_particles, dim)``."""

    levels: Array
    """Integer ladder index per particle, shape ``(n_particles,)``."""

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


@dataclass
class Pipeline:
    """Iterative ST + transport pipeline over a tempered ladder.

    Parameters
    ----------
    n_levels:
        Number of annealing distributions ``0, ..., n_levels - 1``.
    flow:
        Invertible map with ``init`` / ``forward`` / ``inverse`` (e.g. ``NICE``).
    log_prob:
        ``log_prob(level, x) -> Array`` unnormalized log-density for level
        ``level``. ``x`` may be batched ``(n, dim)``; return shape ``(n,)``.
    n_inner:
        ST sweeps inside each ``implement`` call.
    n_local:
        Random-walk MH steps within the current level per particle per sweep.
    local_step_size:
        Multiplier on the Gelman isotropic scale ``2.38 / sqrt(d)``
        (proposal variance ``(2.38² / d) I`` when this is ``1.0``).
    lr:
        Adam learning rate for ``learn_flow``.
    learn_steps:
        Gradient steps per edge per ``learn_flow`` call.
    eta:
        Damping for ``log λ`` updates from occupancy.
    """

    n_levels: int
    flow: Transformation
    log_prob: LogProbFn
    n_inner: int = DEFAULT_INNER_SWEEPS
    n_local: int = DEFAULT_N_LOCAL
    local_step_size: float = DEFAULT_LOCAL_STEP_SIZE
    lr: float = DEFAULT_LEARNING_RATE
    learn_steps: int = DEFAULT_LEARN_STEPS
    eta: float = DEFAULT_ETA
    flows: tuple[Transformation, ...] = field(init=False)
    _optimizer: optax.GradientTransformation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2 (need at least one consecutive pair)")
        if self.n_inner < 1:
            raise ValueError("n_inner must be >= 1")
        if self.learn_steps < 1:
            raise ValueError("learn_steps must be >= 1")
        self.flows = tuple(self.flow for _ in range(self.n_levels - 1))
        self._optimizer = optax.adam(self.lr)

    @property
    def n_edges(self) -> int:
        """Number of consecutive bridges ``i → i+1``."""
        return self.n_levels - 1

    def init_state(self, key: Key, x0: Array) -> PipelineState:
        """Bootstrap state: particles at level 0 from easy samples ``x0``."""
        if x0.ndim != 2:
            raise ValueError("x0 must have shape (n_particles, dim)")
        n_particles = x0.shape[0]
        keys = random.split(key, self.n_edges + 1)
        flow_params = [
            self.flow.init(keys[i])  # type: ignore[attr-defined]
            for i in range(self.n_edges)
        ]
        opt_states = [self._optimizer.init(p) for p in flow_params]
        return PipelineState(
            x=x0,
            levels=jnp.zeros(n_particles, dtype=jnp.int32),
            flow_params=flow_params,
            log_lambda=jnp.zeros(self.n_levels),
            key=keys[-1],
            opt_states=opt_states,
            counts=jnp.zeros(self.n_levels, dtype=jnp.int32),
        )

    def _local_explore(
        self,
        key: Key,
        x: Array,
        level: int,
    ) -> tuple[Key, Array]:
        """Random-walk MH targeting ``π_level`` at fixed ladder index.

        Proposal: ``x' = x + σ ε``, ``ε ~ N(0, I)``, with
        ``σ = local_step_size * 2.38 / sqrt(d)`` so that the default
        (``local_step_size=1``) has variance ``2.38² / d`` per coordinate.
        """
        dim = x.shape[-1]
        step = self.local_step_size * RW_MH_OPTIMAL_SCALE / jnp.sqrt(dim)
        for _ in range(self.n_local):
            key, prop_key, u_key = random.split(key, 3)
            prop = x + step * random.normal(prop_key, x.shape)
            log_alpha = (
                self.log_prob(level, prop[None, :])[0]
                - self.log_prob(level, x[None, :])[0]
            )
            if jnp.log(random.uniform(u_key)) < log_alpha:
                x = prop
        return key, x

    def _jump_ladder(
        self,
        key: Key,
        x: Array,
        level: int,
        proposed: int,
        flow_params: list[Any],
        log_lambda: Array,
    ) -> tuple[Key, Array, int]:
        """Single-particle MH jump between neighboring ladder levels."""
        if proposed == level + 1:
            edge = level
            transform = self.flow.forward
        elif proposed == level - 1:
            edge = proposed
            transform = self.flow.inverse
        else:
            return key, x, level

        y, log_det = transform(flow_params[edge], x[None, :])
        y, log_det = y[0], log_det[0]
        log_alpha = (
            log_lambda[proposed]
            + self.log_prob(proposed, y[None, :])[0]
            - log_lambda[level]
            - self.log_prob(level, x[None, :])[0]
            + log_det
        )
        key, u_key = random.split(key)
        if jnp.log(random.uniform(u_key)) < log_alpha:
            return key, y, proposed
        return key, x, level

    def implement(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """Run ST with frozen flows; accumulate level occupancy."""
        x = state.x
        levels = state.levels
        key = state.key
        counts = jnp.zeros(self.n_levels, dtype=jnp.int32)
        n_particles = int(x.shape[0])
        max_level = self.n_levels - 1
        
        sweeps = range(self.n_inner) if not progress else tqdm(range(self.n_inner), desc="implement (ST)", leave=False)
        for _ in sweeps:
            for p in range(n_particles):
                level = int(levels[p])
                key, new_x = self._local_explore(key, x[p], level)
                x = x.at[p].set(new_x)

                key, dir_key = random.split(key)
                direction = 1 if bool(random.bernoulli(dir_key)) else -1
                proposed = int(jnp.clip(level + direction, 0, max_level))
                key, new_x, new_level = self._jump_ladder(
                    key,
                    x[p],
                    level,
                    proposed,
                    state.flow_params,
                    state.log_lambda,
                )
                x = x.at[p].set(new_x)
                levels = levels.at[p].set(new_level)
            counts = counts + jnp.bincount(levels, length=self.n_levels)

        return replace(state, x=x, levels=levels, key=key, counts=counts)

    def tune_weight(self, state: PipelineState) -> PipelineState:
        """Update ``log λ`` from occupancy counts collected in ``implement``."""
        counts = state.counts.astype(state.log_lambda.dtype)
        # Pseudocounts avoid log(0) when a level was never visited.
        weights = counts + OCCUPANCY_PSEUDOCOUNT
        frac = weights / jnp.sum(weights)
        log_lambda = state.log_lambda - self.eta * jnp.log(frac)
        log_lambda = log_lambda - jnp.mean(log_lambda)
        return replace(state, log_lambda=log_lambda)

    def learn_flow(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """Fit each edge flow with reverse KL on particles at the source level."""
        flow_params = list(state.flow_params)
        opt_states = list(state.opt_states)
        levels_np = state.levels

        edges = range(self.n_edges) if not progress else tqdm(range(self.n_edges), desc="learn_flow", leave=False)
        for edge in edges:
            mask = levels_np == edge
            if not bool(jnp.any(mask)):
                continue
            x_batch = state.x[mask]
            params = flow_params[edge]
            opt_state = opt_states[edge]

            def loss_fn(p: Any, batch: Array = x_batch, e: int = edge) -> Array:
                y, log_det = self.flow.forward(p, batch)
                return jnp.mean(
                    self.log_prob(e, batch) - self.log_prob(e + 1, y) - log_det
                )

            for _ in range(self.learn_steps):
                grads = grad(loss_fn)(params)
                updates, opt_state = self._optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            flow_params[edge] = params
            opt_states[edge] = opt_state

        return replace(state, flow_params=flow_params, opt_states=opt_states)

    def step(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """One outer iteration: implement → tune weight → learn flow."""
        state = self.implement(state, progress=progress)
        state = self.tune_weight(state)
        state = self.learn_flow(state, progress=progress)
        return state

    def run(
        self,
        key: Key,
        x0: Array,
        *,
        n_iterations: int = DEFAULT_OUTER_ITERATIONS,
        progress: bool = True,
        history: list[dict[str, Array]] | None = None,
    ) -> PipelineState:
        """Bootstrap from ``key`` / ``x0``, then repeat ``step`` for ``n_iterations``.

        Equivalent to ``init_state`` then the outer train loop. Afterwards
        ``state.flow_params`` / ``state.log_lambda`` are trained. Pass a list
        as ``history`` to append per-iter ``occupancy`` and ``log_lambda``.
        """
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")
        state = self.init_state(key, x0)

        steps = range(n_iterations) if not progress else tqdm(range(n_iterations), desc="train (outer)", leave=False)
        for _ in steps:
            # Nested bars off: outer bar is enough during ``run``.
            state = self.step(state, progress=False)
            if history is not None:
                total = jnp.maximum(state.counts.sum(), 1)
                history.append(
                    {
                        "occupancy": state.counts / total,
                        "log_lambda": state.log_lambda,
                    }
                )
        return state
