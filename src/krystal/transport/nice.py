"""NICE: Nonlinear Independent Components Estimation (Dinh et al., 2014).

Additive coupling layers are volume-preserving, so ``log|det J| = 0``.
Implemented with Flax ``nn.Module``; the public ``NICE`` wrapper keeps the
``Transformation`` API (``init`` / ``forward`` / ``inverse`` + params pytree).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from flax import linen as nn
from jax import Array


def _split_index(dim: int) -> int:
    if dim < 2:
        raise ValueError("NICE requires dim >= 2 to form additive couplings")
    return dim // 2


class CouplingMLP(nn.Module):
    """MLP used inside one additive coupling (tanh + Xavier)."""

    hidden_dims: tuple[int, ...]
    out_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        kernel_init = nn.initializers.xavier_uniform()
        for width in self.hidden_dims:
            x = nn.Dense(width, kernel_init=kernel_init)(x)
            x = nn.tanh(x)
        return nn.Dense(self.out_dim, kernel_init=kernel_init)(x)


class NICEModule(nn.Module):
    """Flax NICE stack: forward is ``__call__``, reverse is ``inverse``."""

    dim: int
    n_layers: int = 4
    hidden_dims: tuple[int, ...] = (64, 64)

    def setup(self) -> None:
        split = _split_index(self.dim)
        mlps: list[CouplingMLP] = []
        for layer in range(self.n_layers):
            if layer % 2 == 0:
                out_dim = self.dim - split
            else:
                out_dim = split
            mlps.append(CouplingMLP(self.hidden_dims, out_dim))
        self.mlps = mlps

    def __call__(self, x: Array) -> tuple[Array, Array]:
        split = _split_index(self.dim)
        z = x
        for layer_idx, mlp in enumerate(self.mlps):
            a, b = z[..., :split], z[..., split:]
            if layer_idx % 2 == 0:
                b = b + mlp(a)
            else:
                a = a + mlp(b)
            z = jnp.concatenate([a, b], axis=-1)
        log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        return z, log_det

    def inverse(self, y: Array) -> tuple[Array, Array]:
        split = _split_index(self.dim)
        z = y
        for layer_idx in reversed(range(self.n_layers)):
            mlp = self.mlps[layer_idx]
            a, b = z[..., :split], z[..., split:]
            if layer_idx % 2 == 0:
                b = b - mlp(a)
            else:
                a = a - mlp(b)
            z = jnp.concatenate([a, b], axis=-1)
        log_det = jnp.zeros(y.shape[:-1], dtype=y.dtype)
        return z, log_det


@dataclass
class NICE:
    """Transformation-facing wrapper around :class:`NICEModule`.

    ``NICE`` is the architecture config; Flax params are a separate pytree
    from :meth:`init`, passed into :meth:`forward` / :meth:`inverse`.
    """

    dim: int
    n_layers: int = 4
    hidden_dims: tuple[int, ...] = (64, 64)
    _module: NICEModule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._module = NICEModule(
            dim=self.dim,
            n_layers=self.n_layers,
            hidden_dims=self.hidden_dims,
        )

    def init(self, key: Array) -> object:
        """Initialize Flax params (dummy batch of shape ``(1, dim)``)."""
        dummy = jnp.zeros((1, self.dim))
        return self._module.init(key, dummy)["params"]

    def forward(self, params: object, x: Array) -> tuple[Array, Array]:
        """Map ``x → y``; returns ``(y, log|det J|)`` with ``log|det J| = 0``."""
        return self._module.apply({"params": params}, x)

    def inverse(self, params: object, y: Array) -> tuple[Array, Array]:
        """Map ``y → x`` via the exact inverse coupling."""
        return self._module.apply({"params": params}, y, method=self._module.inverse)
