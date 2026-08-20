import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def GetCHatPrime(Y_train, y_train, Y_test):

    AM = LogisticRegression(
        C=np.inf,
        max_iter=1000
    )

    AM.fit(
        Y_train,
        y_train
    )

    c_hat = AM.predict_proba(Y_test)

    # for i in range (5):
    #     p0 = c_hat[i][0]
    #     p1 = c_hat[i][1]
    #     print(f"P(0) = {p0}, P(1) = {p1}")

    y_hat = AM.predict(Y_test)

    eps = 1e-12

    c_hat_clipped = np.clip(
        c_hat,
        eps,
        1 - eps
    )

    c_hat_prime = np.log(
        c_hat_clipped[:, 1] /
        c_hat_clipped[:, 0]
    )

    print(c_hat_prime)

    return c_hat_prime
