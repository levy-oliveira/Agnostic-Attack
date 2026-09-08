from sklearn.linear_model import LogisticRegression


def train_adversary_model(Y_train, y_train):
    """Train the Adversary Model using active-party features."""
    model = LogisticRegression(C=float("inf"), max_iter=1000)
    model.fit(Y_train, y_train)
    return model


def predict_confidence(model, Y_test):
    """Return the positive-class confidence score estimated by the AM."""
    return model.predict_proba(Y_test)[:, 1]


def predict(model, Y_test):
    """Return the predicted class labels of the AM."""
    return model.predict(Y_test)
