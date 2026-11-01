"""Turn the master seed into independent random streams, one per (purpose, index).

Rho and the scenario are not part of the key, so dataset i gets the same base draws at
every rho and in both scenarios (common random numbers).
"""

import numpy as np

# Fixed codes: append new purposes, never renumber, or every stream changes.
PURPOSES = {
    "reference": 0,
    "data": 1,
    "background": 2,
    "explained": 3,
    "model": 4,  # the forest
    "shuffle": 5,
    "background_check": 6,
    "bootstrap": 7,
    "lstm": 8,  # weights, validation split and batch order
    "lstm_seed_check": 9,
    "tuning": 10,
}


def seed_sequence(master_seed, purpose, index=0, replicate=None):
    # replicate: repeated draws on the same dataset (seed check).
    key = (PURPOSES[purpose], index) if replicate is None else (PURPOSES[purpose], index, replicate)
    return np.random.SeedSequence(entropy=master_seed, spawn_key=key)


def rng(master_seed, purpose, index=0, replicate=None):
    """A numpy Generator for one (purpose, index) stream."""
    return np.random.default_rng(seed_sequence(master_seed, purpose, index, replicate))


def int_seed(master_seed, purpose, index=0, replicate=None):
    """A plain int seed, for libraries that want random_state=int."""
    return int(seed_sequence(master_seed, purpose, index, replicate).generate_state(1)[0])
