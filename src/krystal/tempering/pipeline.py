"""CRAFT-style tempering pipeline for Simulated Tempering.

Outer loop order:

1. **Implement** — run ST with current flows; collect level occupancy.
2. **Tune weight** — update ladder weights ``λ`` from occupancy.
3. **Learn flow** — fit ``φ_{i→i+1}`` on the current particle snapshot.

``implement`` is pure JAX (``vmap`` over particles, ``lax.scan`` over sweeps)
so replica chains can be parallelized with ``implement_replicas``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array, grad, lax, random
from jax.random import Key
from jax.tree_util import register_pytree_node_class, tree_map
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

LogProbFn = Callable[[Array | int, Array], Array]


@register_pytree_node_class
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

    def tree_flatten(self):
        children = (
            self.x,
            self.levels,
            self.flow_params,
            self.log_lambda,
            self.key,
            self.opt_states,
            self.counts,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


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
        ``level`` (Python ``int`` or scalar array). ``x`` may be batched
        ``(n, dim)``; return shape ``(n,)``.
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

    def _stack_flow_params(self, flow_params: list[Any]) -> Any:
        """Stack per-edge params into a single pytree with a leading edge axis."""
        return tree_map(lambda *xs: jnp.stack(xs, axis=0), *flow_params)

    def _local_explore(
        self,
        key: Key,
        x: Array,
        level: Array,
    ) -> tuple[Key, Array]:
        """RW-MH at fixed ladder index (single particle; ``vmap``-friendly)."""
        dim = x.shape[-1]
        step = self.local_step_size * RW_MH_OPTIMAL_SCALE / jnp.sqrt(dim)
        log_prob = self.log_prob
        n_local = self.n_local

        def body(carry, _):
            key, x = carry
            key, prop_key, u_key = random.split(key, 3)
            prop = x + step * random.normal(prop_key, x.shape)
            log_alpha = log_prob(level, prop[None, :])[0] - log_prob(level, x[None, :])[0]
            accept = jnp.log(random.uniform(u_key)) < log_alpha
            x = jnp.where(accept, prop, x)
            return (key, x), None

        (key, x), _ = lax.scan(body, (key, x), None, length=n_local)
        return key, x

    def _jump_ladder(
        self,
        key: Key,
        x: Array,
        level: Array,
        stacked_params: Any,
        log_lambda: Array,
    ) -> tuple[Key, Array, Array]:
        """Neighbor MH jump with flow proposal (single particle; ``vmap``-friendly)."""
        max_level = self.n_levels - 1
        flow = self.flow
        log_prob = self.log_prob

        key, dir_key, u_key = random.split(key, 3)
        direction = jnp.where(random.bernoulli(dir_key), 1, -1).astype(level.dtype)
        proposed = jnp.clip(level + direction, 0, max_level)
        going_up = proposed > level
        going_down = proposed < level
        moving = going_up | going_down
        # Forward uses edge ``level``; inverse uses edge ``proposed``.
        edge = jnp.where(going_up, level, proposed)

        params_e = tree_map(lambda p: p[edge], stacked_params)
        y_fwd, ld_fwd = flow.forward(params_e, x[None, :])
        y_inv, ld_inv = flow.inverse(params_e, x[None, :])
        y_fwd, ld_fwd = y_fwd[0], ld_fwd[0]
        y_inv, ld_inv = y_inv[0], ld_inv[0]
        # ``going_up`` is scalar under particle ``vmap``; broadcasts over ``dim``.
        y = jnp.where(going_up, y_fwd, y_inv)
        log_det = jnp.where(going_up, ld_fwd, ld_inv)

        log_alpha = (
            log_lambda[proposed]
            + log_prob(proposed, y[None, :])[0]
            - log_lambda[level]
            - log_prob(level, x[None, :])[0]
            + log_det
        )
        accept = moving & (jnp.log(random.uniform(u_key)) < log_alpha)
        new_x = jnp.where(accept, y, x)
        new_level = jnp.where(accept, proposed, level)
        return key, new_x, new_level

    def _make_sweep(self, stacked_params: Any, log_lambda: Array, n_particles: int):
        """Build one ST sweep: ``vmap`` local explore + jump over particles."""
        n_levels = self.n_levels
        local_explore = self._local_explore
        jump_ladder = self._jump_ladder

        def one_particle(p_key: Key, x: Array, level: Array):
            p_key, x = local_explore(p_key, x, level)
            p_key, x, level = jump_ladder(p_key, x, level, stacked_params, log_lambda)
            return p_key, x, level

        def sweep(carry, _):
            key, x, levels = carry
            keys = random.split(key, n_particles + 1)
            key = keys[0]
            _, x, levels = jax.vmap(one_particle)(keys[1:], x, levels)
            delta = jnp.bincount(levels, length=n_levels)
            return (key, x, levels), delta

        return sweep

    def _run_st(
        self,
        key: Key,
        x: Array,
        levels: Array,
        flow_params: list[Any],
        log_lambda: Array,
        *,
        n_inner: int | None = None,
    ) -> tuple[Key, Array, Array, Array]:
        """Pure ST: ``scan`` over sweeps. Returns ``(key, x, levels, counts)``."""
        n_inner = self.n_inner if n_inner is None else n_inner
        stacked = self._stack_flow_params(flow_params)
        n_particles = int(x.shape[0])
        sweep = self._make_sweep(stacked, log_lambda, n_particles)
        (key, x, levels), deltas = lax.scan(
            sweep, (key, x, levels), None, length=n_inner
        )
        counts = jnp.sum(deltas, axis=0)
        return key, x, levels, counts

    def implement(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """Run ST with frozen flows; accumulate level occupancy."""
        stacked = self._stack_flow_params(state.flow_params)
        n_particles = int(state.x.shape[0])
        sweep = self._make_sweep(stacked, state.log_lambda, n_particles)

        if progress:
            carry = (state.key, state.x, state.levels)
            counts = jnp.zeros(self.n_levels, dtype=jnp.int32)
            sweep_j = jax.jit(sweep)
            for _ in tqdm(range(self.n_inner), desc="implement (ST)", leave=False):
                carry, delta = sweep_j(carry, None)
                counts = counts + delta
            key, x, levels = carry
        else:
            key, x, levels, counts = self._run_st(
                state.key,
                state.x,
                state.levels,
                state.flow_params,
                state.log_lambda,
            )
        return replace(state, x=x, levels=levels, key=key, counts=counts)

    def implement_replicas(
        self,
        state: PipelineState,
        keys: Array,
    ) -> tuple[Array, Array, Array]:
        """Run independent ST chains in parallel from a shared snapshot.

        Parameters
        ----------
        state:
            Trained ensemble (flows / ``λ`` / positions frozen as the shared start).
        keys:
            PRNG keys with shape ``(n_replicas,)`` (typed) or
            ``(n_replicas, 2)`` (legacy ``PRNGKey``).

        Returns
        -------
        xs:
            ``(n_replicas, n_particles, dim)``
        levels:
            ``(n_replicas, n_particles)``
        counts:
            ``(n_replicas, n_levels)``
        """
        # Typed keys: ``(n_replicas,)``; legacy ``PRNGKey``: ``(n_replicas, 2)``.
        if keys.ndim == 0 or keys.ndim > 2 or (keys.ndim == 2 and keys.shape[-1] != 2):
            raise ValueError("keys must have shape (n_replicas,) or (n_replicas, 2)")

        x0 = state.x
        levels0 = state.levels
        flow_params = state.flow_params
        log_lambda = state.log_lambda
        run_st = self._run_st

        def one(key: Key):
            _, x, levels, counts = run_st(key, x0, levels0, flow_params, log_lambda)
            return x, levels, counts

        return jax.vmap(one)(keys)

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

        edges = (
            range(self.n_edges)
            if not progress
            else tqdm(range(self.n_edges), desc="learn_flow", leave=False)
        )
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

        steps = (
            range(n_iterations)
            if not progress
            else tqdm(range(n_iterations), desc="train (outer)", leave=False)
        )
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
