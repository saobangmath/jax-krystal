"""Default hyperparameters and numeric constants for the tempering pipeline."""

# Outer CRAFT-ST loop: implement → tune_weight → learn_flow.
DEFAULT_OUTER_ITERATIONS = 10_000

# ST sweeps inside each ``implement`` call.
DEFAULT_INNER_SWEEPS = 10_000

# Within-level random-walk MH.
DEFAULT_N_LOCAL = 1
# Gelman et al.: proposal ~ N(x, (2.38² / d) Σ), with Σ the (per-level) target
# covariance. Σ = I recovers the isotropic scale 2.38 / sqrt(d).
RW_MH_OPTIMAL_SCALE = 2.38
DEFAULT_LOCAL_STEP_SIZE = 1.0  # multiplier on the Gelman factor 2.38 / sqrt(d)
# Ridge added when forming empirical Σ (and when <2 particles at a level → I).
LOCAL_COV_RIDGE = 1e-6
LOCAL_COV_MIN_POINTS = 2

# Flow training (Adam).
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LEARN_STEPS = 1

# Occupancy → λ update: log λ ← log λ − η log(n_i / N).
DEFAULT_ETA = 1.0

# Pseudocount added to visit counts before normalizing (avoids log 0).
OCCUPANCY_PSEUDOCOUNT = 1e-3
