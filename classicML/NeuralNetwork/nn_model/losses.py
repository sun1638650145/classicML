import numpy as np


def binary_crossentropy(y_pred, y):
    """计算二分交叉熵损失"""
    y_shape = y.shape[0]
    loss = -(np.matmul(y.T, np.log(y_pred)) + np.matmul(1 - y.T, np.log(1 - y_pred))) / y_shape

    return np.squeeze(loss)


def categorical_crossentropy(y_pred, y):
    """计算多分交叉熵损失"""
    y_shape = y.shape[0]
    loss = -np.sum(y * np.log(y_pred)) / y_shape

    return loss


def mean_squared_error(y_pred, y):
    """计算二分类的均方差损失"""
    y_shape = y.shape[0]
    loss = np.sum((y_pred - y) ** 2) / (2 * y_shape)

    return np.squeeze(loss)