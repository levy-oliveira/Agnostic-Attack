import numpy as np


def Half(W_pas, W_act, Y_test, c_hat_prime, b):

    d = W_pas.shape[0]

    # Matriz A
    A = W_pas.reshape(1, d)

    # Pseudoinversa de A
    A_plus = np.linalg.pinv(A)

    # Matriz identidade
    identity = np.eye(d)

    # Vetor de 1s
    ones = np.ones(d)

    # Termo projetado sobre o espaço nulo de A
    null_space_term = (
        0.5 *
        (identity - A_plus @ A) @ ones
    )

    X_hat = []

    for i in range(len(Y_test)):

        # Termo correspondente às features ativas
        active_term = (
            W_act @ Y_test[i]
            + b
        )

        # Equação principal do Half*
        x_hat_i = (
            A_plus @ np.atleast_1d(
                c_hat_prime[i]
                - active_term
            )
            + null_space_term
        )

        X_hat.append(x_hat_i)

    return np.array(X_hat)