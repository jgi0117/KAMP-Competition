"""Validation-only weights for the fifth candidate (no separate grid search)."""

import numpy as np


def inverse_mse_weights(actual, predictions):
    inverse = {name: 1.0 / np.mean((actual - prediction) ** 2)
               for name, prediction in predictions.items()}
    total = sum(inverse.values())
    return {name: value / total for name, value in inverse.items()}
