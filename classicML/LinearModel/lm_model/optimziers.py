from time import time
from .backend import forward, backward
from .backend import compute_loss, compute_accuracy
from .terminal_display import display_verbose


def apply_GradientDescent(beta, grad, learning_rate):
    """基于梯度更新参数"""
    beta -= learning_rate * grad

    return beta


def GradientDescent(x, y, epochs, verbose, beta, learning_rate, loss_function, metrics):
    """梯度下降优化器"""
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

        loss = compute_loss(y_pred, y, loss_function)
        acc = compute_accuracy(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)

    return beta


def NewtonMethod(x, y, epochs, verbose, beta, learning_rate, loss_function, metrics):
    """牛顿法优化器"""
    pass


# alias
GD = GradientDescent
Newton = NewtonMethod