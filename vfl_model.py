import numpy as np
from sklearn.linear_model import LogisticRegression


def train_vfl_model(Y_train, X_train, y_train):
    """Train the VFL logistic-regression model."""
    model = LogisticRegression(C=float("inf"), max_iter=1000)
    model.fit(np.hstack([Y_train, X_train]), y_train)
    return model


def predict_confidence(model, Y_test, X_test):
    """Return the positive-class confidence score of the VFL model."""
    return model.predict_proba(np.hstack([Y_test, X_test]))[:, 1]


def predict(model, Y_test, X_test):
    """Return the predicted class labels of the VFL model."""
    return model.predict(np.hstack([Y_test, X_test]))


def get_weights(model, active_size):
    """Split VFL weights into active/passive parts and return the bias."""
    weights = model.coef_[0]
    bias = model.intercept_[0]
    return weights[:active_size], weights[active_size:], bias