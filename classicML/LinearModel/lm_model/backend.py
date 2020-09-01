import numpy as np
from .losses import mean_squared_error
from .metrics import binary_accuracy


def sigmoid(z):
    """sigmoid激活函数"""
    result = 1 / (1 + np.exp(-z))

    return result


def forward(x, beta):
    """"""
    num_of_x = x.shape[0]
    beta = beta.reshape(-1, 1)

    x_hat = np.c_[x, np.ones((num_of_x, 1))]
    cache = np.matmul(x_hat, beta)
    y_pred = sigmoid(cache)

    return y_pred, x_hat


def backward(y_pred, y, x_hat):
    """"""
    y = y.reshape(-1, 1)
    error = y - y_pred
    grad = (-x_hat * error).sum(0)
    return grad.reshape(-1, 1)


def compute_loss(y_pred, y, loss_function):
    """"""
    if loss_function is None:
        loss = mean_squared_error(y_pred, y)

    return loss


def compute_accuracy(y_pred, y, metrics):
    """"""
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if metrics is None:
        acc = binary_accuracy(y_pred, y)

    return acc