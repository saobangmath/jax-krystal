"""ST + flow experiment on a 2-mode Gaussian mixture.

Target (2D, NICE requires dim >= 2):

    π = (1/3) N(μ_left, 5 I) + (2/3) N(μ_right, 5 I)

with μ_left = (-5, 0), μ_right = (5, 0).

(The query wrote both components as N(0, 5); identical means are not
bimodal, so means are placed at ±5 with variance 5 as specified.)

Easy reference π_0 = N(0, I). Annealed bridge:
    log γ_i = (1 - β_i) log π_0 + β_i log π_target,  β_i = i / (T - 1).

Run:
    PYTHONPATH=src .venv/bin/python examples/bimodal_st.py
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from krystal.tempering import Pipeline
from krystal.transport import NICE

DIM = 2
VAR = 5.0
MEAN_LEFT = jnp.array([-5.0, 0.0])
MEAN_RIGHT = jnp.array([5.0, 0.0])
W_LEFT = 1.0 / 3.0
W_RIGHT = 2.0 / 3.0


def _gauss_logpdf(x, mean, var: float):
    dim = x.shape[-1]
    return -0.5 * (
        dim * jnp.log(2.0 * jnp.pi * var) + jnp.sum((x - mean) ** 2, axis=-1) / var
    )


def target_log_prob(x):
    """log π_target for batched x, shape (n, dim) -> (n,)."""
    a = jnp.log(W_LEFT) + _gauss_logpdf(x, MEAN_LEFT, VAR)
    b = jnp.log(W_RIGHT) + _gauss_logpdf(x, MEAN_RIGHT, VAR)
    return jnp.logaddexp(a, b)


def easy_log_prob(x):
    return _gauss_logpdf(x, jnp.zeros(DIM), 1.0)


def make_log_prob(n_levels: int):
    def log_prob(level: int, x):
        beta = level / max(n_levels - 1, 1)
        return (1.0 - beta) * easy_log_prob(x) + beta * target_log_prob(x)

    return log_prob


def mode_weights(samples_x0: jnp.ndarray) -> tuple[float, float]:
    """Assign by nearest mode on x-axis; return (frac_left, frac_right)."""
    d_left = jnp.sum((samples_x0 - MEAN_LEFT) ** 2, axis=-1)
    d_right = jnp.sum((samples_x0 - MEAN_RIGHT) ** 2, axis=-1)
    left = d_left < d_right
    n = samples_x0.shape[0]
    frac_left = float(jnp.sum(left) / n)
    return frac_left, 1.0 - frac_left


def main() -> None:
    n_levels = 5
    n_particles = 64
    # Moderate budget for a first empirical check (pure-Python ST loop).
    n_outer = 15
    n_inner = 40

    pipe = Pipeline(
        n_levels=n_levels,
        flow=NICE(dim=DIM, n_layers=4, hidden_dims=(32, 32)),
        log_prob=make_log_prob(n_levels),
        n_inner=n_inner,
        n_local=3,
        local_step_size=1.0,
        lr=1e-3,
        learn_steps=5,
        eta=0.5,
    )

    key = random.PRNGKey(0)
    key, k0, k1 = random.split(key, 3)
    x0 = random.normal(k0, (n_particles, DIM))
    state = pipe.init_state(k1, x0)

    print(f"target: {W_LEFT:.3f} N({MEAN_LEFT.tolist()}, {VAR}I) + "
          f"{W_RIGHT:.3f} N({MEAN_RIGHT.tolist()}, {VAR}I)")
    print(f"outer={n_outer}, inner={n_inner}, particles={n_particles}, levels={n_levels}")

    for t in range(n_outer):
        state = pipe.step(state)
        occ = state.counts / jnp.maximum(state.counts.sum(), 1)
        print(
            f"[{t+1:02d}/{n_outer}] occupancy={occ}  "
            f"log_lambda={state.log_lambda}"
        )

    # Extra implement-only collect at target after training.
    state = pipe.implement(state)
    target_level = n_levels - 1
    at_target = state.x[state.levels == target_level]
    print(f"\nparticles at target level {target_level}: {at_target.shape[0]}")

    if at_target.shape[0] == 0:
        print("No particles at target — ladder did not mix enough.")
        return

    frac_left, frac_right = mode_weights(at_target)
    emp_mean = jnp.mean(at_target, axis=0)
    true_mean = W_LEFT * MEAN_LEFT + W_RIGHT * MEAN_RIGHT
    print(f"empirical mode weights: left={frac_left:.3f} right={frac_right:.3f}")
    print(f"true mode weights:      left={W_LEFT:.3f} right={W_RIGHT:.3f}")
    print(f"empirical mean: {emp_mean}")
    print(f"true mean:      {true_mean}")

    # Soft checks printed as pass/fail hints (not hard asserts).
    weight_ok = abs(frac_right - W_RIGHT) < 0.25 and abs(frac_left - W_LEFT) < 0.25
    mean_ok = float(jnp.linalg.norm(emp_mean - true_mean)) < 2.5
    print(f"weight fit (tol 0.25): {'OK' if weight_ok else 'WEAK'}")
    print(f"mean fit (L2 < 2.5):   {'OK' if mean_ok else 'WEAK'}")


if __name__ == "__main__":
    main()
