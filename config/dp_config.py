# '''
# Approximate Differential Privacy (Using Gaussian Mechanism)
# two (ε, δ) privacy parameters
# '''

# '''
# ε - The privacy parameter which can be controlled by the
# data analyst to maintain the trade-off between privacy
# and accuracy
# '''
# EPSILON = 1.0
# '''
# δ - The parameter which tells the probability of privacy
# leak (ε, δ)-differential privacy is known as approximate
# differential privacy
# '''
# DELTA = 1e-5
# CLIP_NORM = 1.0

SENSITIVITY = 1.0

# Fast utility-first
"""
dp_config.py

Approximate Differential Privacy (Gaussian mechanism) configuration.

This file provides sensible defaults tuned for better utility in the
current experimental setup (50 rounds). If you instead want to enforce a
total privacy budget across all rounds, set TOTAL_EPSILON and set
AUTO_DISTRIBUTE=True; the runner will split TOTAL_EPSILON evenly across
rounds (per_round = TOTAL_EPSILON / rounds).

Note: Increasing EPSILON reduces noise (better utility) but weakens privacy.
If you need formal accounting, use an RDP/moments accountant (not included).
"""

# Sensitivity (unused directly in current code but kept for reference)
SENSITIVITY = 1.0

# ---------------------------
# Default per-round DP params
# ---------------------------
# These defaults favor reasonable utility while providing some privacy.
# You can increase EPSILON for better utility or decrease for stronger privacy.
EPSILON = 1.0    # per-round epsilon (larger → less noise per round)
DELTA = 1e-5       # failure probability parameter (small)
CLIP_NORM = 1.0    # L2 clip applied to updates (larger → allow bigger updates)

# ---------------------------
# Optional total-budget mode
# ---------------------------
# If you set TOTAL_EPSILON and enable AUTO_DISTRIBUTE, the runner will
# compute per_round = TOTAL_EPSILON / rounds and use that epsilon per round.
TOTAL_EPSILON = 50.0
# Enable auto-distribution of TOTAL_EPSILON across rounds (per_round = TOTAL_EPSILON / rounds)
AUTO_DISTRIBUTE = False
