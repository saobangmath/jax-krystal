"""NICE invertibility and diagonal-scaling checks."""

import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

from krystal.transport import NICE, Transformation


def test_nice_is_transformation():
    model = NICE(dim=4, n_layers=2, hidden_dims=(8,))
    assert isinstance(model, Transformation)


def test_nice_inverse_roundtrip():
    key = random.PRNGKey(0)
    model = NICE(dim=4, n_layers=4, hidden_dims=(16, 16))
    params = model.init(key)
    # Nontrivial final scaling so log|det| is not identically zero.
    params = tree_map(
        lambda x: x,
        {
            **params,
            "log_s": jnp.array([0.2, -0.1, 0.3, -0.05]),
        },
    )
    x = random.normal(random.PRNGKey(1), (8, 4))

    y, log_det_f = model.forward(params, x)
    x_rec, log_det_b = model.inverse(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x_rec, x, atol=1e-5)
    expected = jnp.sum(params["log_s"])
    assert jnp.allclose(log_det_f, expected, atol=1e-5)
    assert jnp.allclose(log_det_b, -expected, atol=1e-5)
    assert jnp.allclose(log_det_f + log_det_b, 0.0, atol=1e-5)


def test_nice_init_scaling_is_identity():
    model = NICE(dim=3, n_layers=2, hidden_dims=(8,))
    params = model.init(random.PRNGKey(0))
    assert jnp.allclose(params["log_s"], 0.0)
    x = random.normal(random.PRNGKey(1), (4, 3))
    _, log_det = model.forward(params, x)
    assert jnp.allclose(log_det, 0.0)
