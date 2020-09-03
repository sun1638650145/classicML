import numpy as np

from .losses import mean_squared_error, log_likelihood
from .metrics import binary_accuracy


def sigmoid(z):
    """sigmoid激活函数"""
    result = 1 / (1 + np.exp(-z))

    return result


def forward(x, beta):
    """前向传播"""
    num_of_x = x.shape[0]
    beta = beta.reshape(-1, 1)

    x_hat = np.c_[x, np.ones((num_of_x, 1))]
    cache = np.matmul(x_hat, beta)
    y_pred = sigmoid(cache)

    return y_pred, x_hat


def backward(y_pred, y, x_hat):
    """反向传播"""
    y = y.reshape(-1, 1)
    error = y - y_pred
    grad = (-x_hat * error).sum(0)
    return grad.reshape(-1, 1)


def solve_hessian_matrix(y_pred, x_hat):
    """求解海森矩阵(二阶导数)"""
    P = np.eye(x_hat.shape[0]) * y_pred * (1 - y_pred)

    return np.dot(np.dot(x_hat.T, P), x_hat)


def compute_loss(y_pred, y, beta, x_hat, loss_function):
    """计算损失值"""
    if loss_function is None:
        loss = log_likelihood(y_pred, beta, x_hat)
    elif loss_function in (mean_squared_error, 'mean_squared_error'):
        loss = mean_squared_error(y_pred, y)
    else:
        raise Exception('请检查输入的损失函数')

    return loss


def compute_metric(y_pred, y, metrics):
    """计算评估值"""
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if metrics is None:
        acc = binary_accuracy(y_pred, y)
    elif metrics in (binary_accuracy, 'binary_accuracy'):
        acc = binary_accuracy(y_pred, y)
    else:
        raise Exception('请检查输入的评估函数')

    return acc