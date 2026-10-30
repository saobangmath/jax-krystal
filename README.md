# Krystal

**Krystal** is a [JAX](https://jax.readthedocs.io/) package for **optimal transport neural networks** with **tempering algorithms**.

It aims to combine learned transport maps (and related OT-inspired neural models) with tempering / annealing schedules so sampling, training, and bridging between distributions can be done in a stable, differentiable way on accelerators.

## Motivation

Optimal transport (OT) gives a principled way to move mass between probability measures. Neural OT methods learn maps or couplings with deep networks, but training and sampling can be brittle when source and target are far apart or when the landscape is multi-modal.

**Tempering** (annealing) introduces a path of intermediate distributions that gradually interpolate between an easy reference and a hard target. Krystal’s goal is to fold tempering into OT neural workflows so that:

- transport is learned along tempered bridges, not only end-to-end;
- gradients and particles remain better behaved under large distributional gap;
- schedules and OT objectives share one JAX-native stack (JIT, vmap, pmap/shard_map).

## Scope (planned)

| Area | Focus |
|------|--------|
| **Neural OT** | Learned maps / potentials / couplings in JAX (Flax / Equinox-friendly) |
| **Tempering** | Temperature / geometric / annealing schedules for training & sampling |
| **Bridges** | Tempered OT paths between source ↔ target |
| **Accelerators** | Pure JAX: composable transforms, reproducible PRNG, GPU/TPU ready |

Exact APIs will land as the package grows; this README describes the intended direction.

## Status

Early scaffolding. Editable install works; tempering has minimal schedule helpers; OT modules are placeholders.

## Project layout

```text
src/krystal/
  ot/           # neural OT (placeholder)
  tempering/    # annealing schedules
  utils/        # shared helpers (placeholder)
tests/          # smoke / unit tests
```

## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

PyPI publish (`pip install krystal`) is not set up yet.

## Quick start

```python
import krystal as kt

schedule = kt.tempering.geometric(32, start=1.0, end=1e-3)
# linear schedule also available: kt.tempering.linear(32)
```

Neural OT maps and tempered training loops will land in later iterations.

## Design principles

- **JAX-first** — functional transforms, explicit PRNG, accelerator-friendly code
- **OT + tempering as first-class** — schedules and transport objectives composed, not bolted on
- **Research-friendly** — small, readable building blocks over opaque frameworks

## Roadmap (high level)

1. Project layout, dependencies, and typing ← *in progress*
2. Core tempering schedules and path utilities
3. Neural OT building blocks (maps / losses)
4. Joint training loops with tempered bridges
5. Tutorials and benchmarks

## Citation

If you use this work in research, a citation entry will be added here once a paper or preprint is available.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and ideas welcome. Keep changes small so we can iterate on the structure safely.
