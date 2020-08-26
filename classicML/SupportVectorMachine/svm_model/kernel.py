import numpy as np


def linear_kernel(x_i, x_j):
    """
        线性核函数
        kappa(x_i, x_j) = x_i^T * x_j
    """
    return np.dot(x_j, x_i.T)


def polynomial_kernel(x_i, x_j, degree):
    """
        多项式核函数
        kappa(x_i, x_j) = (x_i^T * x_j)^d
    """
    return np.power(np.dot(x_j, x_i.T), degree)


def rbf_kernel(x_i, x_j, gamma):
    """
        高斯核函数
        kappa(x_i, x_j) = exp(-gamma * (||x_i - x_j||^2)/2*sigma^2)
    """
    return np.exp(-gamma * np.sum(np.power(x_j - x_i, 2), axis=1))