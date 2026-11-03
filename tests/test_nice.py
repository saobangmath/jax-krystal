"""NICE invertibility and volume-preserving checks."""

import jax.numpy as jnp
from jax import random

from krystal.transport import NICE, Transformation


def test_nice_is_transformation():
    model = NICE(dim=4, n_layers=2, hidden_dims=(8,))
    assert isinstance(model, Transformation)


def test_nice_inverse_roundtrip():
    key = random.PRNGKey(0)
    model = NICE(dim=4, n_layers=4, hidden_dims=(16, 16))
    params = model.init(key)
    x = random.normal(random.PRNGKey(1), (8, 4))

    y, log_det_f = model.forward(params, x)
    x_rec, log_det_b = model.inverse(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x_rec, x, atol=1e-5)
    assert jnp.allclose(log_det_f, 0.0)
    assert jnp.allclose(log_det_b, 0.0)
