"""CRAFT-style tempering pipeline for Simulated Tempering.

One outer iteration (``Pipeline.step``) has three stages:

1. **Implement** — run ST with frozen flows; count how often each level is visited.
2. **Tune weight** — update ladder weights ``λ`` from those visit counts.
3. **Learn flow** — fit ``φ_{i→i+1}`` on the particles currently at level ``i``.

Where things live:

- ``context.py`` — ``STContext``, the read-only inputs shared by every sweep.
- ``kernels.py`` — the per-particle moves (random-walk MH, lifted ladder jump)
  and ``st_sweep``, which ``vmap``s them over all particles.
- ``state.py`` — ``PipelineState``, the data carried between stages.
- this file — the orchestration: ``implement`` scans ``st_sweep`` over
  ``n_inner`` sweeps; ``tune_weight`` / ``learn_flow`` are the two updates.

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
from jax import Array, lax, random, value_and_grad
from jax.random import Key
from tqdm.auto import tqdm

from krystal.tempering.constants import (
    DEFAULT_DIVERGENCE,
    DEFAULT_ETA,
    DEFAULT_INNER_SWEEPS,
    DEFAULT_LEARN_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOCAL_STEP_SIZE,
    DEFAULT_N_LOCAL,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_PLATEAU_SMOOTHING,
    DEFAULT_PLATEAU_TOL,
    LOCAL_COV_MIN_POINTS,
    LOCAL_COV_RIDGE,
    OCCUPANCY_PSEUDOCOUNT,
)
from krystal.tempering.context import LogProbFn, STContext
from krystal.tempering.kernels import st_sweep
from krystal.tempering.state import PipelineState
from krystal.tempering.stopping import PlateauStopper
from krystal.transport.base import Transformation


_DIVERGENCES = ("symmetric", "reverse")


def edge_loss(log_q: Array, log_p: Array, divergence: str = DEFAULT_DIVERGENCE) -> Array:
    """Flow objective on a batch of points ``y`` drawn from ``q`` (the pushed-forward source).

    ``log_q`` and ``log_p`` are per-sample log-densities at the same points, and
    may be unnormalized.

    - ``"reverse"``: ``KL(q ‖ p) = E_q[log q − log p]``.
    - ``"symmetric"``: ``KL(q ‖ p) + KL(p ‖ q)``. The forward term is estimated
      by self-normalized importance sampling with weights ``w ∝ p / q``, which
      gives ``Σ softmax(log w) · log w`` for ``log w = log p − log q``.

    A constant offset in either density (e.g. an unknown normalizer) cancels
    in the symmetric objective, so its absolute value is meaningful.
    """
    if divergence not in _DIVERGENCES:
        raise ValueError(f"divergence must be one of {_DIVERGENCES}")
    log_w = log_p - log_q
    reverse = jnp.mean(-log_w)
    if divergence == "reverse":
        return reverse
    forward = jnp.sum(jax.nn.softmax(log_w) * log_w)
    return reverse + forward


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
        Multiplier on the Gelman factor ``2.38 / sqrt(d)``. With per-level
        covariance ``Σ`` on :class:`PipelineState`, the proposal is
        ``N(x, (2.38² / d) Σ)`` (``Σ = I`` until ``update_local_cov``).
    lr:
        Adam learning rate for ``learn_flow``.
    learn_steps:
        Gradient steps per edge per ``learn_flow`` call.
    eta:
        Damping for ``log λ`` updates from occupancy.
    divergence:
        Flow objective: ``"symmetric"`` (default) or ``"reverse"``. See :func:`edge_loss`.
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
    divergence: str = DEFAULT_DIVERGENCE
    flows: tuple[Transformation, ...] = field(init=False)
    _optimizer: optax.GradientTransformation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2 (need at least one consecutive pair)")
        if self.n_inner < 1:
            raise ValueError("n_inner must be >= 1")
        if self.learn_steps < 1:
            raise ValueError("learn_steps must be >= 1")
        if self.divergence not in _DIVERGENCES:
            raise ValueError(f"divergence must be one of {_DIVERGENCES}")
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
        local_cov = jnp.broadcast_to(
            jnp.eye(dim, dtype=x0.dtype), (self.n_levels, dim, dim)
        ).copy()
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
            local_cov=local_cov,
            flow_losses=jnp.full(self.n_edges, jnp.nan, dtype=x0.dtype),
        )

    def _stack_flow_params(self, flow_params: list[Any]) -> Any:
        """Stack per-edge params into one pytree with a leading edge axis.

        Lets a traced edge index pick params inside ``vmap``: ``p[edge]``.
        """
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *flow_params)

    def _context(
        self, flow_params: list[Any], log_lambda: Array, local_cov: Array
    ) -> STContext:
        """Freeze the current flows / ``λ`` / ``Σ`` plus settings for the kernels."""
        return STContext(
            flow=self.flow,
            log_prob=self.log_prob,
            n_levels=self.n_levels,
            step_size=self.local_step_size,
            n_local=self.n_local,
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
        ctx = self._context(flow_params, log_lambda, local_cov)

        def one_sweep(carry, _):
            key, x, levels, directions = carry
            key, x, levels, directions, visits = st_sweep(
                key, x, levels, directions, ctx
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
            state.local_cov,
        )
        return replace(
            state, x=x, levels=levels, directions=directions, key=key, counts=counts
        )

    def update_local_cov(self, state: PipelineState) -> PipelineState:
        """Refresh per-level ``Σ`` from the current particle cloud.

        For each ladder level with at least ``LOCAL_COV_MIN_POINTS`` particles,
        set ``Σ`` to the empirical covariance (+ ridge). Levels with too few
        particles keep ``I``. Call this after burn-in / collect when dimensions
        look correlated, then continue with ``implement``.
        """
        x = state.x
        levels = state.levels
        dim = x.shape[-1]
        eye = jnp.eye(dim, dtype=x.dtype)
        ridge = LOCAL_COV_RIDGE * eye
        min_points = LOCAL_COV_MIN_POINTS
        n_levels = self.n_levels

        def cov_for_level(level: Array) -> Array:
            mask = levels == level
            n = jnp.sum(mask)
            x_sum = jnp.sum(jnp.where(mask[:, None], x, 0.0), axis=0)
            mean = x_sum / jnp.maximum(n, 1)
            xc = jnp.where(mask[:, None], x - mean, 0.0)
            cov = (xc.T @ xc) / jnp.maximum(n - 1, 1) + ridge
            return jnp.where(n >= min_points, cov, eye)

        local_cov = jax.vmap(cov_for_level)(jnp.arange(n_levels))
        return replace(state, local_cov=local_cov)

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
        flow_losses = state.flow_losses
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
                # Density of y under the pushed-forward source, and the target at y.
                log_q = self.log_prob(e, batch) - log_det
                log_p = self.log_prob(e + 1, y)
                return edge_loss(log_q, log_p, self.divergence)

            for _ in range(self.learn_steps):
                loss, grads = value_and_grad(loss_fn)(params)
                updates, opt_state = self._optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            # Loss at the parameters entering the last step (pre-update).
            flow_losses = flow_losses.at[edge].set(loss)
            flow_params[edge] = params
            opt_states[edge] = opt_state

        return replace(
            state,
            flow_params=flow_params,
            opt_states=opt_states,
            flow_losses=flow_losses,
        )

    def step(
        self,
        state: PipelineState,
        *,
        progress: bool = False,
    ) -> PipelineState:
        """One outer iteration: implement → tune weight → learn flow."""
        state = self.implement(state)
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
        patience: int | None = None,
        tol: float = DEFAULT_PLATEAU_TOL,
        smoothing: float = DEFAULT_PLATEAU_SMOOTHING,
    ) -> PipelineState:
        """Bootstrap from ``key`` / ``x0``, then repeat ``step`` up to ``n_iterations``.

        Equivalent to ``init_state`` then the outer train loop. Afterwards
        ``state.flow_params`` / ``state.log_lambda`` are trained. Pass a list
        as ``history`` to append per-iter ``occupancy``, ``log_lambda``, and
        ``flow_loss`` (per-edge reverse-KL loss from ``learn_flow``).

        ``n_iterations`` is a cap. If ``patience`` is set, training stops early
        once the smoothed mean flow loss has not improved by more than ``tol``
        (relative) for ``patience`` consecutive iterations; see
        :class:`~krystal.tempering.stopping.PlateauStopper`. With
        ``patience=None`` (default) it always runs ``n_iterations``.
        """
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")
        stopper = (
            PlateauStopper(patience=patience, tol=tol, smoothing=smoothing)
            if patience is not None
            else None
        )
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
                        "flow_loss": state.flow_losses,
                    }
                )
            if stopper is not None and stopper.update(state.flow_losses):
                break
        return state
