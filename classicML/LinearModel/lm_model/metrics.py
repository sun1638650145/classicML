import numpy as np


def binary_accuracy(y_pred, y):
    """"""
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    acc = np.mean(np.equal(y_pred, y))

    return acc