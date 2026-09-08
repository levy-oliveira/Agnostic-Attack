import numpy as np


def mean_squared_error(y_true, y_pred):
    """Return the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def feature_mean_squared_error(X_true, X_pred):
    """Return one MSE value per feature."""
    return np.mean((X_true - X_pred) ** 2, axis=0)
