"""Pipeline: ladder wiring and CRAFT-ST stage smoke tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax import random

from krystal.tempering import Pipeline, PipelineState
from krystal.transport import NICE


def _gaussian_log_prob(x, mean, var):
    dim = x.shape[-1]
    return -0.5 * (
        dim * jnp.log(2.0 * jnp.pi * var)
        + jnp.sum((x - mean) ** 2, axis=-1) / var
    )


def _annealed_log_prob(n_levels: int, target_mean: float = 2.0, target_var: float = 0.5):
    """Geometric bridge between N(0, I) and N(m, v I)."""

    def log_prob(level: int, x):
        mix = level / max(n_levels - 1, 1)
        log_easy = _gaussian_log_prob(x, 0.0, 1.0)
        log_target = _gaussian_log_prob(x, target_mean, target_var)
        return (1.0 - mix) * log_easy + mix * log_target

    return log_prob


def _tiny_pipeline(**kwargs) -> Pipeline:
    n_levels = kwargs.pop("n_levels", 3)
    defaults = dict(
        n_levels=n_levels,
        flow=NICE(dim=2, n_layers=2, hidden_dims=(8,)),
        log_prob=_annealed_log_prob(n_levels),
        n_inner=5,
        learn_steps=1,
        lr=1e-2,
        eta=0.5,
    )
    defaults.update(kwargs)
    return Pipeline(**defaults)


def test_pipeline_builds_one_flow_per_edge():
    pipe = _tiny_pipeline(n_levels=5)
    assert pipe.n_levels == 5
    assert pipe.n_edges == 4
    assert len(pipe.flows) == 4


def test_pipeline_rejects_single_level():
    with pytest.raises(ValueError, match="n_levels"):
        _tiny_pipeline(n_levels=1)


def test_init_state_at_level_zero():
    pipe = _tiny_pipeline()
    key = random.PRNGKey(0)
    x0 = random.normal(key, (16, 2))
    state = pipe.init_state(random.PRNGKey(1), x0)
    assert isinstance(state, PipelineState)
    assert state.x.shape == (16, 2)
    assert state.levels.shape == (16,)
    assert jnp.all(state.levels == 0)
    assert state.directions.shape == (16,)
    assert jnp.all(state.directions == 1)
    assert len(state.flow_params) == pipe.n_edges
    assert state.log_lambda.shape == (pipe.n_levels,)
    assert state.local_cov.shape == (pipe.n_levels, 2, 2)
    assert jnp.allclose(state.local_cov[0], jnp.eye(2))


def test_implement_updates_counts_and_preserves_shape():
    pipe = _tiny_pipeline(n_inner=8)
    key = random.PRNGKey(0)
    x0 = random.normal(key, (12, 2))
    state = pipe.init_state(random.PRNGKey(1), x0)
    state = pipe.implement(state)
    assert state.x.shape == (12, 2)
    assert state.counts.shape == (pipe.n_levels,)
    assert int(state.counts.sum()) == 12 * pipe.n_inner
    assert jnp.all((state.levels >= 0) & (state.levels < pipe.n_levels))
    assert state.directions.shape == (12,)
    assert jnp.all((state.directions == 1) | (state.directions == -1))


def test_tune_weight_changes_log_lambda():
    pipe = _tiny_pipeline(n_inner=10, eta=1.0)
    key = random.PRNGKey(0)
    state = pipe.init_state(random.PRNGKey(1), random.normal(key, (20, 2)))
    state = pipe.implement(state)
    before = state.log_lambda
    state = pipe.tune_weight(state)
    assert state.log_lambda.shape == before.shape
    # With non-uniform occupancy, λ should move.
    assert not jnp.allclose(state.log_lambda, before)


def test_learn_flow_runs_when_particles_at_level_zero():
    pipe = _tiny_pipeline(n_inner=2, learn_steps=2)
    key = random.PRNGKey(0)
    state = pipe.init_state(random.PRNGKey(1), random.normal(key, (16, 2)))
    # All at level 0 → edge 0 should update (Flax params pytree).
    leaf_before = jax.tree_util.tree_leaves(state.flow_params[0])[0]
    state = pipe.learn_flow(state)
    leaf_after = jax.tree_util.tree_leaves(state.flow_params[0])[0]
    assert not jnp.allclose(leaf_before, leaf_after)


def test_step_and_short_run_smoke():
    pipe = _tiny_pipeline(n_inner=4, learn_steps=1)
    key = random.PRNGKey(0)
    x0 = random.normal(key, (8, 2))
    state = pipe.init_state(random.PRNGKey(1), x0)
    state = pipe.step(state)
    assert state.counts.sum() > 0
    history: list = []
    state = pipe.run(random.PRNGKey(2), x0, n_iterations=2, progress=False, history=history)
    assert state.x.shape == (8, 2)
    assert len(history) == 2
    assert history[0]["occupancy"].shape == (pipe.n_levels,)
    assert len(state.flow_params) == pipe.n_edges


def test_implement_replicas_parallel_collect():
    pipe = _tiny_pipeline(n_inner=4)
    key = random.PRNGKey(0)
    x0 = random.normal(key, (10, 2))
    state = pipe.init_state(random.PRNGKey(1), x0)
    n_replicas = 3
    keys = random.split(random.PRNGKey(2), n_replicas)
    xs, levels, directions, counts = pipe.implement_replicas(state, keys)
    assert xs.shape == (n_replicas, 10, 2)
    assert levels.shape == (n_replicas, 10)
    assert directions.shape == (n_replicas, 10)
    assert counts.shape == (n_replicas, pipe.n_levels)
    assert int(counts.sum()) == n_replicas * 10 * pipe.n_inner
    assert jnp.all((levels >= 0) & (levels < pipe.n_levels))
    assert jnp.all((directions == 1) | (directions == -1))
    # Shared start + different keys → not all replicas identical.
    assert not jnp.allclose(xs[0], xs[1])


def test_lifted_jump_flips_at_hot_boundary():
    """At level 0 with V=-1, boundary reflection flips V without moving."""
    pipe = _tiny_pipeline(n_inner=1, n_levels=3)
    key = random.PRNGKey(0)
    x0 = random.normal(key, (4, 2))
    state = pipe.init_state(random.PRNGKey(1), x0)
    state = replace(state, directions=-jnp.ones(4, dtype=jnp.int32))
    # One jump attempt from the boundary downward → all should flip to +1, stay at 0.
    stacked = pipe._stack_flow_params(state.flow_params)
    keys = random.split(random.PRNGKey(2), 4)

    def one(k, x, level, direction):
        _, x2, level2, direction2 = pipe._jump_ladder(
            k, x, level, direction, stacked, state.log_lambda
        )
        return x2, level2, direction2

    xs, levels, directions = jax.vmap(one)(
        keys, state.x, state.levels, state.directions
    )
    assert jnp.all(levels == 0)
    assert jnp.all(directions == 1)
    assert jnp.allclose(xs, state.x)


def test_update_local_cov_estimates_correlation():
    pipe = _tiny_pipeline(n_inner=2)
    key = random.PRNGKey(0)
    # Correlated cloud all at level 0: x1 ≈ x0.
    z = random.normal(key, (64, 1))
    x0 = jnp.concatenate([z, z], axis=-1)
    state = pipe.init_state(random.PRNGKey(1), x0)
    state = pipe.update_local_cov(state)
    cov0 = state.local_cov[0]
    # Off-diagonal should be clearly positive (near 1 for this cloud).
    assert cov0[0, 1] > 0.5
    # Empty levels stay I.
    assert jnp.allclose(state.local_cov[1], jnp.eye(2), atol=1e-5)
    # Non-identity Σ still runs implement.
    state = pipe.implement(state)
    assert state.x.shape == (64, 2)
