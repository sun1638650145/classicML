import numpy as np


def he_normal(layer_dim, seed):
    """
        He正态分布初始化器
        w~n(0, sqrt(2/n_in))，其中n_in为对应输入层的神经元个数
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    np.random.seed(seed)

    parameters = {}
    num_of_layers = len(layer_dim)
    for layer in range(num_of_layers - 1):
        w = np.random.randn(layer_dim[layer + 1], layer_dim[layer]) * np.sqrt(2/layer_dim[layer])
        b = np.zeros((1, layer_dim[layer + 1]))

        parameters['w' + str(layer + 1)] = w
        parameters['b' + str(layer + 1)] = b

    return parameters


def xavier(layer_dim, seed):
    """
        Xavier正态分布初始化器
        w~n(0, sqrt(2/(n_in+n_out)))，其中n_in为对应输入层的神经元个数，n_out为对应输出层的神经元个数
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    np.random.seed(seed)

    parameters = {}
    num_of_layers = len(layer_dim)
    for layer in range(num_of_layers - 1):
        w = np.random.randn(layer_dim[layer + 1], layer_dim[layer]) * np.sqrt(2/(layer_dim[layer] + layer_dim[layer + 1]))
        b = np.zeros((1, layer_dim[layer + 1]))

        parameters['w' + str(layer + 1)] = w
        parameters['b' + str(layer + 1)] = b

    return parameters


def adam_initializer(parameters):
    """对Adam参数初始化"""
    L = int(len(parameters) / 2)

    m = {}
    v = {}

    for i in range(1, L + 1):
        m['d_w' + str(i)] = np.zeros(parameters['w' + str(i)].shape)
        m['d_b' + str(i)] = np.zeros(parameters['b' + str(i)].shape)
        v['d_w' + str(i)] = np.zeros(parameters['w' + str(i)].shape)
        v['d_b' + str(i)] = np.zeros(parameters['b' + str(i)].shape)

    return m, v


def rbf_initializer(hidden_units, seed):
    """对RBF网络参数初始化"""
    np.random.seed(seed)

    parameters = {}

    parameters['w'] = np.zeros([1, hidden_units])
    parameters['b'] = np.zeros([1, 1])
    parameters['c'] = np.random.rand(hidden_units, 2)  # 神经元对应的中心坐标
    parameters['beta'] = np.random.randn(1, hidden_units)  # 高斯径向基函数的beta

    return parameters


# alias
glorot = xavier
