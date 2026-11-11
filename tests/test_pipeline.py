"""Pipeline: ladder wiring and CRAFT-ST stage smoke tests."""

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
    assert len(state.flow_params) == pipe.n_edges
    assert state.log_lambda.shape == (pipe.n_levels,)


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
    xs, levels, counts = pipe.implement_replicas(state, keys)
    assert xs.shape == (n_replicas, 10, 2)
    assert levels.shape == (n_replicas, 10)
    assert counts.shape == (n_replicas, pipe.n_levels)
    assert int(counts.sum()) == n_replicas * 10 * pipe.n_inner
    assert jnp.all((levels >= 0) & (levels < pipe.n_levels))
    # Shared start + different keys → not all replicas identical.
    assert not jnp.allclose(xs[0], xs[1])
