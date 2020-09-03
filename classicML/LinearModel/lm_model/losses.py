import numpy as np


def mean_squared_error(y_pred, y):
    """均方差"""
    y_shape = y.shape[0]
    loss = np.sum((y_pred - y) ** 2 / (2 * y_shape))

    return np.squeeze(loss)


def log_likelihood(y, beta, x_hat):
    """对数似然损失"""
    y = y.reshape(-1, 1)
    beta = beta.reshape(-1, 1)

    loss = -y * np.matmul(x_hat, beta) + np.log(1 + np.exp(np.matmul(x_hat, beta)))

    return np.sum(loss)