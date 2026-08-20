import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from adversary_model import GetCHatPrime

def Half(W_pas,W_act,Y_train, y_train, Y_test, b, ):
    k = 2
    d = 5

    J = np.array([
        [-1, 1]
    ])

    A = J @ W_pas

    A_plus = np.linalg.pinv(A)

    identity = np.eye(d)

    ones = np.ones(d)

    X_hat = (
        A_plus @ (
            GetCHatPrime(Y_train, y_train, Y_test)
            - J @ W_act @ Y_test[i]
            - J @ b
        )
        +
        0.5 * (
            identity
            - A_plus @ A
        ) @ ones
    )

    return X_hat