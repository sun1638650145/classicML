from .losses import *
from .metrics import *


def ReLU(z):
    """线性整流激活函数"""
    result = np.maximum(0, z)

    return result


def sigmoid(z):
    """sigmoid激活函数"""
    result = 1 / (1 + np.exp(-z))

    return result


def softmax(z):
    """softmax激活函数"""
    z -= np.max(z)  # 为了避免溢出，一般减去最大值
    result = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    return result


def forward_one_layer(x, w, b, activation):
    """计算单层的前向传播值"""
    z = np.matmul(x, w.T) + b

    if activation in ('relu', 'ReLU'):
        a = ReLU(z)
    elif activation == 'sigmoid':
        a = sigmoid(z)
    elif activation in ('softmax', 'Softmax'):
        a = softmax(z)

    cache = (x, w, b, z)
    return a, cache


def forward(x, parameters):
    """计算前向传播"""
    L = int(len(parameters) / 2)
    caches = []
    a = x
    for i in range(1, L):
        # 取出参数值
        w = parameters['w' + str(i)]
        b = parameters['b' + str(i)]
        a_temp = a

        a, cache = forward_one_layer(a_temp, w, b, 'relu')
        caches.append(cache)

    w_out = parameters['w' + str(L)]
    b_out = parameters['b' + str(L)]

    # 输出层使用不同的激活函数
    if w_out.shape[0] == 1:
        y_pred, cache = forward_one_layer(a, w_out, b_out, 'sigmoid')
    else:
        y_pred, cache = forward_one_layer(a, w_out, b_out, 'softmax')

    caches.append(cache)

    return y_pred, caches


def ReLU_backward(y, z):
    """
        线性整流激活函数
        relu函数的导数在大于0的区间应为f'=1，但是会出现收敛到分类问题随机概率上，
        个人感觉应该类似relu导数小于0的区间的f'=0造成的神经死亡现象类似，
        随着更新，大部分有效神经元的输出为恒为1，成为一个线性的神经元，相当于不参与计算了，
        暂还不能完全解释；
        但是使用f原值就可以避免
    """
    d_z = np.asarray(y)
    d_z[z <= 0] = 0

    return d_z


def sigmoid_backward(y_pred, y, z):
    """sigmoid激活函数"""
    a = sigmoid(z)
    d_z = (y - y_pred) * a * (1 - a)

    return d_z


def softmax_backward(y, z):
    """
        softmax激活函数
        https://blog.csdn.net/qq_38032064/article/details/90599547?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
    """
    a = softmax(z)
    d_z = a - y

    return d_z


def backward_one_layer(y_pred, y, cache, activation):
    (x, w, b, z) = cache
    y_shape = y.shape[0]

    if activation in ('relu', 'ReLU'):
        d_z = ReLU_backward(y, z)
    elif activation == 'sigmoid':
        d_z = sigmoid_backward(y_pred, y, z)
    elif activation in ('softmax', 'Softmax'):
        d_z = softmax_backward(y, z)

    d_w = np.matmul(d_z.T, x) / y_shape
    d_b = np.sum(d_z, axis=0, keepdims=True) / y_shape
    # 计算梯度 梯度就是关于连接权重w的导数
    d_pred = np.matmul(d_z, w)

    return d_pred, d_w, d_b


def backward(y_pred, y, caches):
    """计算反向传播"""
    grad = {}
    L = len(caches)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if y.shape[1] == 1:
        # 二分类的时候，y为0无法计算梯度
        # 将原来的y映射到[-2,0)和(0,2]区间上
        d_a = -(y / y_pred - (1 - y) / (1 - y_pred))
        d_a_pred, d_w, d_b = backward_one_layer(y_pred, d_a, caches[L-1], 'sigmoid')
    else:
        d_a_pred, d_w, d_b = backward_one_layer(y_pred, y, caches[L-1], 'softmax')

    grad['d_a' + str(L)] = d_a_pred
    grad['d_w' + str(L)] = d_w
    grad['d_b' + str(L)] = d_b

    for i in range(L - 1, 0, -1):
        # 逆向计算
        d_a_pred, d_w, d_b = backward_one_layer(y_pred, grad['d_a' + str(i + 1)], caches[i-1], 'relu')

        grad['d_a' + str(i)] = d_a_pred
        grad['d_w' + str(i)] = d_w
        grad['d_b' + str(i)] = d_b

    return grad


def rbf_forward(x, parameters):
    """计算RBF的前向传播"""
    num_of_x, attr_of_x = x.shape

    w = parameters['w']
    b = parameters['b']
    c = parameters['c']
    beta = parameters['beta']

    units = c.shape[0]
    rho = np.zeros((num_of_x, units))
    x_ci = np.zeros((num_of_x, units))
    # 高斯径向基函数
    # rho(x, c_i) = e^{-beta||x-c_i||^2)}
    for unit in range(units):
        x_ci[:, unit] = np.linalg.norm(x - c[[unit], ], axis=1) ** 2
        rho[:, unit] = np.exp(-beta[0, unit] * x_ci[:, unit])

    y_pred = np.matmul(rho, w.T) + b
    cache = (rho, x_ci, w, beta)

    return y_pred, cache


def rbf_backward(y_pred, y, cache):
    """计算RBF的反向传播"""
    grad = {}
    (rho, x_ci, w, beta) = cache
    y_shape = y.shape[0]

    d_y = y_pred - y  # 这里不能叫d_y 就是预测值和真实值之间的误差
    d_w = np.matmul(d_y.T, rho) / y_shape
    d_b = np.sum(d_y, axis=0, keepdims=True) / y_shape

    d_rho = np.matmul(y_pred, w)
    d_beta = np.sum(d_rho * rho * (-x_ci), axis=0, keepdims=True) / y_shape

    grad['d_w'] = d_w
    grad['d_b'] = d_b
    grad['d_beta'] = d_beta

    return grad


def compute_loss(y_pred, y, loss_function, model=None):
    """计算loss"""
    if loss_function is None:
        if model is None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] == 1:
                loss = binary_crossentropy(y_pred, y)
            else:
                loss = categorical_crossentropy(y_pred, y)
        else:
            loss = mean_squared_error(y_pred, y)
    else:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if loss_function in ('binary_crossentropy', binary_crossentropy):
            loss = binary_crossentropy(y_pred, y)
        elif loss_function in ('ce', 'categorical_crossentropy', categorical_crossentropy):
            loss = categorical_crossentropy(y_pred, y)
        elif loss_function in ('mse', 'mean_squared_error', mean_squared_error):
            loss = mean_squared_error(y_pred, y)
        else:
            raise Exception("请检查输入的损失函数")

    return loss


def compute_accuracy(y_pred, y, metrics):
    """计算准确度"""
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if metrics in ('binary_accuracy', binary_accuracy):
        acc = binary_accuracy(y_pred, y)
    elif metrics in ('categorical_accuracy', categorical_accuracy):
        acc = categorical_accuracy(y_pred, y)
    else:
        # 没有写就自动对推理
        if y.shape[1] == 1:
            acc = binary_accuracy(y_pred, y)
        else:
            acc = categorical_accuracy(y_pred, y)

    return acc