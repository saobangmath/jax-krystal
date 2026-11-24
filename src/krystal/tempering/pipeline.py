"""CRAFT-style tempering pipeline for Simulated Tempering.

One outer iteration (``Pipeline.step``) has three stages:

1. **Implement** — run ST with frozen flows; count how often each level is visited.
2. **Tune weight** — update ladder weights ``λ`` from those visit counts.
3. **Learn flow** — fit ``φ_{i→i+1}`` on the particles currently at level ``i``.

Where things live:

- ``context.py`` — ``STContext`` (unchanging settings) and ``SweepInputs``
  (frozen arrays for one ``implement``, including ``chol(local_cov)``).
- ``kernels.py`` — the per-particle moves (random-walk MH, lifted ladder jump)
  and ``st_sweep``, which ``vmap``s them over all particles.
- ``state.py`` — ``PipelineState``, the data that changes between stages
  (including Welford stats and derived ``local_cov``).
- this file — the orchestration: ``implement`` scans ``st_sweep`` over
  ``n_inner`` sweeps; ``update_local_cov`` / ``tune_weight`` / ``learn_flow``
  are the updates.

Ladder jumps use a **lifted / non-reversible** kernel: each particle carries a
persistent direction ``V ∈ {+1, -1}``, always proposes ``I → I + V`` (with flow),
keeps ``V`` on accept, and flips ``V`` on MH reject or at a ladder boundary
(ST analogue of DEO persistence from non-reversible PT).

``implement`` is pure JAX (``vmap`` over particles, ``lax.scan`` over sweeps)
so independent chains can be run in parallel by wrapping ``implement`` in ``jax.vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array, grad, lax, random
from jax.random import Key
from tqdm.auto import tqdm

from krystal.tempering.constants import (
    DEFAULT_ETA,
    DEFAULT_INNER_SWEEPS,
    DEFAULT_LEARN_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_LOCAL,
    DEFAULT_OUTER_ITERATIONS,
    LOCAL_COV_MIN_POINTS,
    LOCAL_COV_RIDGE,
    OCCUPANCY_PSEUDOCOUNT,
)
from krystal.tempering.context import LogProbFn, STContext, SweepInputs
from krystal.tempering.kernels import st_sweep
from krystal.tempering.state import LocalCovState, PipelineState
from krystal.transport.base import Transformation


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
        Proposal scale is the Gelman factor ``2.38 / sqrt(d)`` with per-level
        covariance ``Σ`` on :class:`PipelineState` (``Σ = I`` until
        ``update_local_cov``).
    lr:
        Adam learning rate for ``learn_flow``.
    learn_steps:
        Joint gradient steps per ``learn_flow`` call; each updates every edge
        that has particles at its source level.
    eta:
        Damping for ``log λ`` updates from occupancy.
    """

    n_levels: int
    flow: Transformation
    log_prob: LogProbFn
    n_inner: int = DEFAULT_INNER_SWEEPS
    n_local: int = DEFAULT_N_LOCAL
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
        dim = x0.shape[1]
        return PipelineState(
            x=x0,
            levels=jnp.zeros(n_particles, dtype=jnp.int32),
            # Start climbing from the hot end (level 0).
            directions=jnp.ones(n_particles, dtype=jnp.int32),
            flow_params=flow_params,
            log_lambda=jnp.zeros(self.n_levels),
            key=keys[-1],
            opt_states=opt_states,
            counts=jnp.zeros(self.n_levels, dtype=jnp.int32),
            local_cov=LocalCovState.init(self.n_levels, dim, dtype=x0.dtype),
        )

    def _stack_flow_params(self, flow_params: list[Any]) -> Any:
        """Stack per-edge params into one pytree with a leading edge axis.

        Lets a traced edge index pick params inside ``vmap``: ``p[edge]``.
        """
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *flow_params)

    def _st_context(self) -> STContext:
        """Unchanging settings for the kernels (same for every outer iteration)."""
        return STContext(
            flow=self.flow,
            log_prob=self.log_prob,
            n_levels=self.n_levels,
            n_local=self.n_local,
        )

    def _sweep_inputs(
        self, flow_params: list[Any], log_lambda: Array, local_cov: Array
    ) -> SweepInputs:
        """Snapshot changing arrays for one ``implement``; derive ``chol(Σ)``."""
        return SweepInputs(
            stacked_params=self._stack_flow_params(flow_params),
            log_lambda=log_lambda,
            cov_chols=jnp.linalg.cholesky(local_cov),
        )

    def _run_st(
        self,
        key: Key,
        x: Array,
        levels: Array,
        directions: Array,
        flow_params: list[Any],
        log_lambda: Array,
        local_cov: Array,
    ) -> tuple[Key, Array, Array, Array, Array]:
        """Scan ``st_sweep`` for ``n_inner`` sweeps (pure function of its inputs).

        Returns ``(key, x, levels, directions, counts)`` where ``counts`` sums
        the per-sweep level visits, shape ``(n_levels,)``.
        """
        ctx = self._st_context()
        sweep = self._sweep_inputs(flow_params, log_lambda, local_cov)

        def one_sweep(carry, _):
            key, x, levels, directions = carry
            key, x, levels, directions, visits = st_sweep(
                key, x, levels, directions, ctx, sweep
            )
            return (key, x, levels, directions), visits

        (key, x, levels, directions), visits = lax.scan(
            one_sweep, (key, x, levels, directions), None, length=self.n_inner
        )
        return key, x, levels, directions, visits.sum(axis=0)

    def implement(self, state: PipelineState) -> PipelineState:
        """Run lifted ST with frozen flows; accumulate level occupancy."""
        key, x, levels, directions, counts = self._run_st(
            state.key,
            state.x,
            state.levels,
            state.directions,
            state.flow_params,
            state.log_lambda,
            state.local_cov.cov,
        )
        return replace(
            state, x=x, levels=levels, directions=directions, key=key, counts=counts
        )

    def update_local_cov(self, state: PipelineState) -> PipelineState:
        """Merge the current particle cloud into per-level Welford stats.

        For each level with at least ``LOCAL_COV_MIN_POINTS`` particles, fold
        that batch into :class:`~krystal.tempering.state.LocalCovState` with
        Chan's parallel update (no particle history stored). Then set::

            cov = M2 / (n - 1) + ridge · I   if n >= 2 else I

        Sparse levels leave their Welford stats unchanged. Called after
        ``implement`` in :meth:`step`.
        """
        x = state.x
        levels = state.levels
        prev = state.local_cov
        dim = x.shape[-1]
        eye = jnp.eye(dim, dtype=x.dtype)
        ridge = LOCAL_COV_RIDGE * eye
        min_points = LOCAL_COV_MIN_POINTS
        n_levels = self.n_levels

        def merge_level(level: Array) -> tuple[Array, Array, Array, Array]:
            mask = levels == level
            n_b = jnp.sum(mask).astype(x.dtype)
            x_sum = jnp.sum(jnp.where(mask[:, None], x, 0.0), axis=0)
            mean_b = x_sum / jnp.maximum(n_b, 1.0)
            xc = jnp.where(mask[:, None], x - mean_b, 0.0)
            m2_b = xc.T @ xc

            n_a = prev.n[level]
            mean_a = prev.mean[level]
            m2_a = prev.m2[level]

            n = n_a + n_b
            delta = mean_b - mean_a
            # Chan merge; if n_a == 0 this reduces to the batch stats.
            mean = mean_a + delta * (n_b / jnp.maximum(n, 1.0))
            m2 = m2_a + m2_b + jnp.outer(delta, delta) * (
                n_a * n_b / jnp.maximum(n, 1.0)
            )

            use_batch = n_b >= min_points
            n_new = jnp.where(use_batch, n, n_a)
            mean_new = jnp.where(use_batch, mean, mean_a)
            m2_new = jnp.where(use_batch, m2, m2_a)
            cov = jnp.where(
                n_new >= 2.0,
                m2_new / jnp.maximum(n_new - 1.0, 1.0) + ridge,
                eye,
            )
            return n_new, mean_new, m2_new, cov

        n, mean, m2, cov = jax.vmap(merge_level)(jnp.arange(n_levels))
        return replace(
            state,
            local_cov=LocalCovState(n=n, mean=mean, m2=m2, cov=cov),
        )

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
        """Fit all edge flows jointly with reverse KL on particles at each source level.

        Each of the ``learn_steps`` steps takes one gradient of the *sum* of the
        per-edge losses, then applies Adam only to edges that have particles.
        Inactive edges keep their params and optimizer state untouched.
        """
        flow_params = list(state.flow_params)
        opt_states = list(state.opt_states)

        active: list[int] = []
        batches: list[Array] = []
        for edge in range(self.n_edges):
            mask = state.levels == edge
            if bool(jnp.any(mask)):
                active.append(edge)
                batches.append(state.x[mask])
        if not active:
            return state

        def total_loss(params_list: list[Any]) -> Array:
            total = jnp.zeros(())
            for edge, params, batch in zip(active, params_list, batches):
                y, log_det = self.flow.forward(params, batch)
                total = total + jnp.mean(
                    self.log_prob(edge, batch) - self.log_prob(edge + 1, y) - log_det
                )
            return total

        params_list = [flow_params[edge] for edge in active]
        steps = (
            range(self.learn_steps)
            if not progress
            else tqdm(range(self.learn_steps), desc="learn_flow", leave=False)
        )
        for _ in steps:
            grads = grad(total_loss)(params_list)
            for i, edge in enumerate(active):
                updates, opt_states[edge] = self._optimizer.update(
                    grads[i], opt_states[edge], params_list[i]
                )
                params_list[i] = optax.apply_updates(params_list[i], updates)

        for edge, params in zip(active, params_list):
            flow_params[edge] = params

        return replace(state, flow_params=flow_params, opt_states=opt_states)

    def step(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """One outer iteration: implement → update Σ → tune weight → learn flow."""
        state = self.implement(state)
        state = self.update_local_cov(state)
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
