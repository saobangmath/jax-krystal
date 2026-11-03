"""Map-style transformation interface (NF-first).

A transformation transports mass between neighboring tempering levels.
Normalizing flows implement this with an exact inverse and log|det J|.
Diffusion backends can be added later behind a richer proposal API if needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Transformation(Protocol):
    """Invertible transport: ``x → y`` with density correction."""

    def forward(self, params: object, x: Array) -> tuple[Array, Array]:
        """Map ``x`` to ``y`` and return ``(y, log_abs_det_jacobian)``."""
        ...

    def inverse(self, params: object, y: Array) -> tuple[Array, Array]:
        """Map ``y`` to ``x`` and return ``(x, log_abs_det_jacobian)`` of the inverse."""
        ...
