"""Minimal smoke tests: install/import and tiny schedule helpers."""

import jax.numpy as jnp

import krystal as kt


def test_import_and_version():
    assert isinstance(kt.__version__, str)
    assert kt.__version__


def test_linear_schedule_shape():
    schedule = kt.tempering.linear(8)
    assert schedule.shape == (8,)
    assert jnp.isfinite(schedule).all()


def test_geometric_schedule_positive():
    schedule = kt.tempering.geometric(5, start=1.0, end=1e-2)
    assert schedule.shape == (5,)
    assert (schedule > 0).all()
