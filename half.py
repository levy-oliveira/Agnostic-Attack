import numpy as np


def Half(W_pas, W_act, Y_test, c_hat_prime, b):

    d = W_pas.shape[0]

    # W_pas como matriz linha
    A = W_pas.reshape(1, d)

    # Pseudoinversa de Moore-Penrose
    A_plus = np.linalg.pinv(A)

    # Projeção sobre o espaço nulo de A
    null_space_projection = (
        np.eye(d) - A_plus @ A
    )

    # Termo Half*
    null_space_term = (
        0.5 * null_space_projection @ np.ones(d)
    )

    X_hat = []

    for i in range(len(Y_test)):

        # Contribuição das features ativas
        active_term = W_act @ Y_test[i] + b

        # Parte atribuída às features passivas
        passive_logit = c_hat_prime[i] - active_term

        # Reconstrução Half*
        x_hat_i = (
            A_plus @ np.atleast_1d(passive_logit)
            + null_space_term
        )

        X_hat.append(x_hat_i)

    return np.asarray(X_hat)