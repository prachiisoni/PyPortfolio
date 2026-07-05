"""
Generate random portfolios.
"""

import numpy as np


def generate_random_weights(num_assets):
    """
    Generate random portfolio weights.

    The weights always sum to 1.
    """

    weights = np.random.random(num_assets)

    weights /= np.sum(weights)

    return weights