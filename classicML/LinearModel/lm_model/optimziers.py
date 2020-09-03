from time import time

import numpy as np

from .backend import forward, backward
from .backend import compute_loss, compute_metric
from .backend import solve_hessian_matrix
from .terminal_display import display_verbose


def apply_GradientDescent(beta, grad, learning_rate):
    """基于梯度更新参数"""
    beta -= learning_rate * grad

    return beta


def GradientDescent(x, y, epochs, verbose, beta, learning_rate, loss_function, metrics):
    """梯度下降优化器
    Parameters
    ----------
    x : numpy.ndarray or array-like
        特征数据
    y : numpy.ndarray or array-like
        标签
    epochs : int, default=1
        训练的轮数
    verbose : bool, default=True, optional
        显示日志信息
    beta : numpy.ndarray
        逻辑回归的参数矩阵
    learning_rate : float
        学习率
    loss_function : str or function
        损失函数
    metrics : str or function
        评估函数

    Returns
    -------
    beta : numpy.ndarray
        逻辑回归的参数矩阵
    """
    ETD = time()
    for epoch in range(epochs):
        # 每轮开始的时间
        starting_time = time()
        # 前向传播
        y_pred, x_hat = forward(x, beta)
        # 反向传播
        grad = backward(y_pred, y, x_hat)
        # 更新参数
        beta = apply_GradientDescent(beta, grad, learning_rate)

        loss = compute_loss(y_pred, y, beta, x_hat, loss_function)
        acc = compute_metric(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
    print()

    return beta


def apply_NewtonMethod(beta, grad, hessian):
    """更新牛顿法参数"""
    beta -= np.matmul(np.linalg.inv(hessian), grad)

    return beta


def NewtonMethod(x, y, epochs, verbose, beta, loss_function, metrics):
    """牛顿法优化器
    Parameters
    ----------
    x : numpy.ndarray or array-like
        特征数据
    y : numpy.ndarray or array-like
        标签
    epochs : int, default=1
        训练的轮数
    verbose : bool, default=True, optional
        显示日志信息
    beta : numpy.ndarray
        逻辑回归的参数矩阵
    loss_function : str or function
        损失函数
    metrics : str or function
        评估函数

    Returns
    -------
    beta : numpy.ndarray
        逻辑回归的参数矩阵
    """
    ETD = time()
    for epoch in range(epochs):
        # 每轮开始的时间
        starting_time = time()
        # 前向传播
        y_pred, x_hat = forward(x, beta)
        # 反向传播
        grad = backward(y_pred, y, x_hat)
        # 求解海森矩阵
        hessian = solve_hessian_matrix(y_pred, x_hat)
        # 更新参数
        beta = apply_NewtonMethod(beta, grad, hessian)

        loss = compute_loss(y_pred, y, beta, x_hat, loss_function)
        acc = compute_metric(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
    print()

    return beta


# alias
GD = GradientDescent
Newton = NewtonMethod