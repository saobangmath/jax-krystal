"""Default hyperparameters and numeric constants for the tempering pipeline."""

# Outer CRAFT-ST loop: implement → tune_weight → learn_flow.
DEFAULT_OUTER_ITERATIONS = 10_000

# ST sweeps inside each ``implement`` call.
DEFAULT_INNER_SWEEPS = 10_000

# Within-level random-walk MH.
DEFAULT_N_LOCAL = 1
DEFAULT_LOCAL_STEP_SIZE = 0.5

# Flow training (Adam).
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LEARN_STEPS = 1

# Occupancy → λ update: log λ ← log λ − η log(n_i / N).
DEFAULT_ETA = 1.0

# Pseudocount added to visit counts before normalizing (avoids log 0).
OCCUPANCY_PSEUDOCOUNT = 1e-3
