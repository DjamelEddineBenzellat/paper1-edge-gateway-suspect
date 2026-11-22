"""Seeds for every random draw, from (run seed, purpose, round, device, step).

Each draw gets its own stream, so results do not depend on the order in
which Ray runs the gateways.
"""
import numpy as np

INIT, SAMPLE, NOISE, GROUPS, SHUFFLE, DETECT, CANARY = 1, 2, 3, 4, 5, 6, 7  # DETECT: DL-37; CANARY: DL-38
SIZE_AWARE = 8  # DL-44: the Monte Carlo of the size-aware gateway check


def seed_int(*keys):
    """A 63-bit integer seed derived from the keys."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1, dtype=np.uint64)[0] >> 1)


def rng(*keys):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in keys]))
