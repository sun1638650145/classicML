import numpy as np
from time import time
from .terminal_display import display_verbose
from .backend import forward, backward, rbf_forward, rbf_backward
from .backend import compute_loss, compute_accuracy
from .initializers import adam_initializer


def apply_GradientDescent(parameters, grad, learning_rate):
    """基于梯度更新参数"""
    L = int(len(parameters) / 2)

    for i in range(1, L + 1):
        parameters['w' + str(i)] -= learning_rate * grad['d_w' + str(i)]
        parameters['b' + str(i)] -= learning_rate * grad['d_b' + str(i)]

    return parameters


def apply_Adam(parameters, grad, learning_rate, beta_1, beta_2, epsilon, m, v, epoch):
    """
        Adam更新参数 参考论文的算法1
        https://arxiv.org/abs/1412.6980
    """
    L = int(len(parameters) / 2)

    for i in range(1, L + 1):
        m['d_w' + str(i)] = beta_1 * m['d_w' + str(i)] + (1 - beta_1) * grad['d_w' + str(i)]
        m['d_b' + str(i)] = beta_1 * m['d_b' + str(i)] + (1 - beta_1) * grad['d_b' + str(i)]

        v['d_w' + str(i)] = beta_2 * v['d_w' + str(i)] + (1 - beta_2) * (grad['d_w' + str(i)] ** 2)
        v['d_b' + str(i)] = beta_2 * v['d_b' + str(i)] + (1 - beta_2) * (grad['d_b' + str(i)] ** 2)

        m_w_correct = m['d_w' + str(i)] / (1 - np.power(beta_1, epoch))
        m_b_correct = m['d_b' + str(i)] / (1 - np.power(beta_1, epoch))

        v_w_correct = v['d_w' + str(i)] / (1 - np.power(beta_2, epoch))
        v_b_correct = v['d_b' + str(i)] / (1 - np.power(beta_2, epoch))

        parameters['w' + str(i)] -= learning_rate * m_w_correct / np.sqrt(v_w_correct + epsilon)
        parameters['b' + str(i)] -= learning_rate * m_b_correct / np.sqrt(v_b_correct + epsilon)

    return parameters, m, v


def apply_RBF(parameters, grad, learning_rate):
    """基于梯度更新参数"""
    parameters['w'] -= learning_rate * grad['d_w']
    parameters['b'] -= learning_rate * grad['d_b']
    parameters['beta'] -= learning_rate * grad['d_beta']

    return parameters


def GradientDescent(x, y, epochs, verbose, parameters, learning_rate, loss_function, metrics):
    """梯度下降优化器"""
    loss_list = []
    acc_list = []

    ETD = time()
    for epoch in range(epochs):
        # 每轮开始时间
        starting_time = time()
        # 前向传播
        y_pred, caches = forward(x, parameters)
        # 反向传播
        grad = backward(y_pred, y, caches)
        # 更新参数
        parameters = apply_GradientDescent(parameters, grad, learning_rate)

        loss = compute_loss(y_pred, y, loss_function)
        acc = compute_accuracy(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
        loss_list.append(loss)
        acc_list.append(acc)

    print()

    return parameters, loss_list, acc_list


def StochasticGradientDescent(x, y, epochs, verbose, parameters, learning_rate, loss_function, metrics, seed):
    """随机梯度下降优化器"""
    np.random.seed(seed)

    loss_list = []
    acc_list = []

    num_of_features = x.shape[0]

    ETD = time()
    for epoch in range(epochs):
        # 每轮开始时间
        starting_time = time()
        # 随机选择样本
        random_index = np.random.randint(0, num_of_features)
        # 用于更新参数的y_pred_one的前向传播
        y_pred_one, caches = forward(x[[random_index], :], parameters)
        grad = backward(y_pred_one, y[[random_index], :], caches)

        parameters = apply_GradientDescent(parameters, grad, learning_rate)
        # 更新参数后计算损失的y_pred，y_pred_one维度和y不一致不便于计算损失
        y_pred, _ = forward(x, parameters)

        loss = compute_loss(y_pred, y, loss_function)
        acc = compute_accuracy(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
        loss_list.append(loss)
        acc_list.append(acc)

    print()

    return parameters, loss_list, acc_list


def Adam(x, y, epochs, verbose, parameters, loss_function, metrics, seed, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
    """自适应矩估计优化器"""
    np.random.seed(seed)

    loss_list = []
    acc_list = []

    num_of_features = x.shape[0]

    # 对Adam进行初始化
    m, v = adam_initializer(parameters)

    ETD = time()
    for epoch in range(epochs):
        # 每轮开始时间
        starting_time = time()
        # 随机选择样本
        random_index = np.random.randint(0, num_of_features)
        # 用于更新参数的y_pred_one
        y_pred_one, caches = forward(x[[random_index], :], parameters)
        grad = backward(y_pred_one, y[[random_index], :], caches)

        parameters, m, v = apply_Adam(parameters, grad, learning_rate, beta_1, beta_2, epsilon, m, v, epoch+1)
        # 更新参数后计算损失的y_pred
        y_pred, _ = forward(x, parameters)

        loss = compute_loss(y_pred, y, loss_function)
        acc = compute_accuracy(y_pred, y, metrics)

        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
        loss_list.append(loss)
        acc_list.append(acc)

    print()

    return parameters, loss_list, acc_list


def RBFOptimizer(x, y, epochs, verbose, parameters, learning_rate):
    """RBF网络优化器"""
    loss_list = []
    acc_list = []

    ETD = time()
    for epoch in range(epochs):
        # 每轮开始时间
        starting_time = time()
        # 前向传播
        y_pred, cache = rbf_forward(x, parameters)
        # 反向传播
        grad = rbf_backward(y_pred, y, cache)
        # 更新参数
        parameters = apply_RBF(parameters, grad, learning_rate)

        loss = compute_loss(y_pred, y, loss_function=None, model='RBF')
        acc = compute_accuracy(y_pred, y, 'binary_accuracy')
        if verbose:
            display_verbose(epoch, epochs, loss, acc, starting_time, ETD)
        loss_list.append(loss)
        acc_list.append(acc)

    print()

    return parameters, loss_list, acc_list


# alias
GD = GradientDescent
SGD = StochasticGradientDescent
