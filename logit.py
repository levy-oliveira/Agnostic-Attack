import numpy as np


def probability_to_logit(probability):
    """Convert probabilities to logits with numerical clipping."""
    probability = np.clip(probability, 1e-15, 1 - 1e-15)
    return np.log(probability / (1 - probability))
