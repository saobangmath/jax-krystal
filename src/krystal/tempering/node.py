"""Simulated tempering node: ladder index + internal state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    """One ST node on the tempered path.

    Attributes
    ----------
    temperature_ladder:
        Integer index ``i`` into the shared temperature ladder
        (levels ``0, ..., T``). The actual β / temperature values live on
        the ladder container, not on each node.
    internal_state:
        Node-local state (e.g. configuration ``x``, or a broader pytree).
    """

    temperature_ladder: int
    internal_state: Any
