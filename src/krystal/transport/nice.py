"""NICE: Nonlinear Independent Components Estimation (Dinh et al., 2014).

Additive coupling layers are volume-preserving, so ``log|det J| = 0``.
That keeps MH corrections simple while wiring flows into simulated tempering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random


def _mlp_init(
    key: Array,
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int],
) -> list[tuple[Array, Array]]:
    """Initialize an MLP as a list of ``(weight, bias)``."""
    dims = [in_dim, *hidden_dims, out_dim]
    params: list[tuple[Array, Array]] = []
    keys = random.split(key, len(dims) - 1)
    for k, din, dout in zip(keys, dims[:-1], dims[1:], strict=True):
        # Glorot-style scaling for tanh networks
        limit = jnp.sqrt(6.0 / (din + dout))
        w = random.uniform(k, (din, dout), minval=-limit, maxval=limit)
        b = jnp.zeros((dout,))
        params.append((w, b))
    return params


def _mlp_apply(params: Sequence[tuple[Array, Array]], x: Array) -> Array:
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


def _split_index(dim: int) -> int:
    if dim < 2:
        raise ValueError("NICE requires dim >= 2 to form additive couplings")
    return dim // 2


@dataclass
class NICE:
    """Stack of additive coupling layers (NICE).

    Even layers transform the second half given the first; odd layers do the
    reverse split — the usual alternating partition.
    """

    dim: int
    n_layers: int = 4
    hidden_dims: tuple[int, ...] = (64, 64)

    def init(self, key: Array) -> list[list[tuple[Array, Array]]]:
        """Sample parameters for every coupling network ``m``."""
        split = _split_index(self.dim)
        # Even layers: m(first) -> second half; odd: m(second) -> first half
        in_out = [
            (split, self.dim - split) if layer % 2 == 0 else (self.dim - split, split)
            for layer in range(self.n_layers)
        ]
        keys = random.split(key, self.n_layers)
        return [
            _mlp_init(k, din, dout, self.hidden_dims)
            for k, (din, dout) in zip(keys, in_out, strict=True)
        ]

    def forward(self, params: object, x: Array) -> tuple[Array, Array]:
        layers = params
        assert isinstance(layers, list)
        split = _split_index(self.dim)
        z = x
        for layer_idx, m_params in enumerate(layers):
            if layer_idx % 2 == 0:
                a, b = z[..., :split], z[..., split:]
                b = b + _mlp_apply(m_params, a)
                z = jnp.concatenate([a, b], axis=-1)
            else:
                a, b = z[..., :split], z[..., split:]
                a = a + _mlp_apply(m_params, b)
                z = jnp.concatenate([a, b], axis=-1)
        log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        return z, log_det

    def inverse(self, params: object, y: Array) -> tuple[Array, Array]:
        layers = params
        assert isinstance(layers, list)
        split = _split_index(self.dim)
        z = y
        for layer_idx, m_params in reversed(list(enumerate(layers))):
            if layer_idx % 2 == 0:
                a, b = z[..., :split], z[..., split:]
                b = b - _mlp_apply(m_params, a)
                z = jnp.concatenate([a, b], axis=-1)
            else:
                a, b = z[..., :split], z[..., split:]
                a = a - _mlp_apply(m_params, b)
                z = jnp.concatenate([a, b], axis=-1)
        log_det = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        return z, log_det
