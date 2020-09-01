import numpy as np


def mean_squared_error(y_pred, y):
    """均方差"""
    y_shape = y.shape[0]
    loss = np.sum((y_pred - y) ** 2 / (2 * y_shape))

    return np.squeeze(loss)