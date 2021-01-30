"""classicML的优化器."""
import os
from time import time

import numpy as np

if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.activations import relu
    from classicML.backend.cc.activations import sigmoid
    from classicML.backend.cc.activations import softmax
else:
    from classicML.backend.python.activations import relu
    from classicML.backend.python.activations import sigmoid
    from classicML.backend.python.activations import softmax

from classicML.backend.python.callbacks import History

from classicML.backend.python.utils import ProgressBar

if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.ops import cc_calculate_error as calculate_error
    from classicML.backend.cc.ops import cc_clip_alpha as clip_alpha
    from classicML.backend.cc.ops import cc_select_second_alpha as select_second_alpha
else:
    from classicML.backend.python.ops import calculate_error
    from classicML.backend.python.ops import clip_alpha
    from classicML.backend.python.ops import select_second_alpha


def _get_optimizer_parameters(args, kwargs):
    """获取优化器的额外参数.

    Arguments:
        args: *args元组.
        kwargs: **kwargs字典.

    Returns:
        额外的参数列表.
    """
    parameters = ['verbose', 'loss', 'metric', 'callbacks']
    for index, arg in enumerate(args):
        parameters[index] = arg

    for kwarg in kwargs:
        for index, parameter in enumerate(parameters):
            if str(kwarg) == parameter:
                parameters[index] = kwargs[kwarg]

    return parameters


def _record_callbacks(callbacks, loss_value, metric_value):
    """记录callbacks数据.

    Arguments:
        callbacks: list, callbacks列表.
        loss_value: float, 当前的损失值.
        metric_value: float, 当前的评估值.
    """
    for callback in callbacks:
        if isinstance(callback, History):
            callback(loss_value, metric_value)
        else:
            pass


class Optimizer(object):
    """优化器的基类.

    Attributes:
        name: str, default=None,
            优化器的名称.
        _progressbar: classicML.backend.python.utils.ProgressBar,
            进度条.
    """
    def __init__(self, name=None):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
        """
        self.name = name
        self._progressbar = None

    def __call__(self, x, y, epochs, parameters, *args, **kwargs):
        """函数实现.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            parameters: numpy.ndarray, 模型的参数矩阵.
        """
        return self.run(x, y, epochs, parameters, *args, **kwargs)

    def run(self, x, y, epochs, parameters, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            parameters: numpy.ndarray, 模型的参数矩阵.
        """
        raise NotImplementedError

    def _update_parameters(self, parameters, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            parameters: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.
        """
        raise NotImplementedError


class StochasticGradientDescent(Optimizer):
    """随机梯度下降优化器.

    Notes:
        - 如果想固定随机种子, 实现复现的话,
          请在模型实例化的时候将随机种子置为一个常整数.
    """

    def __init__(self, name='stochastic_gradient_descent', learning_rate=1e-2):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
            learning_rate: float, default=1e-2,
                优化器的学习率.
        """
        super(StochasticGradientDescent, self).__init__(name=name)
        self.learning_rate = learning_rate

    def run(self, x, y, epochs, parameters, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            parameters: numpy.ndarray, 模型的参数矩阵.
            args:
                verbose: bool, 显示日志信息.
                loss: str, classicML.losses.Loss 实例,
                    模型使用的损失函数.
                metric: str, classicML.metrics.Metric 实例,
                    模型使用的评估函数.
                callbacks: list,
                    模型训练过程的中间数据记录器.
        Returns:
            模型的参数矩阵.
        """
        verbose, loss, metric, callbacks = _get_optimizer_parameters(args, kwargs)
        num_of_features = x.shape[0]

        if verbose is True:
            # 实例化进度条
            self._progressbar = ProgressBar(epochs, loss, metric)

        for epoch in range(1, epochs + 1):
            current = time()
            # 随机选取样本
            random_index = np.random.randint(0, num_of_features)
            # 前向传播
            y_pred, caches = self.forward(x[[random_index], :], parameters)
            # 反向传播
            # Notes: 此时y_true是个scalar
            grad = self.backward(y_pred, np.asarray([y[random_index]]), caches)
            # 更新参数
            parameters = self._update_parameters(parameters, grad)
            # 以下的操作是为了更好可视化的, 不参与实际优化
            if self._progressbar or callbacks is not None:
                # 由于每次只更新一个样本导致预测值的形状和标签不一致,
                # 因此需要多做一次前向传播
                y_pred, _ = self.forward(x, parameters)

                loss_value = loss(y_pred, y)
                metric_value = metric(y_pred, y)

                # 进度条显示
                if self._progressbar:
                    self._progressbar(epoch, current, loss_value, metric_value)
                # 记录callbacks
                if callbacks:
                    _record_callbacks(callbacks, loss_value, metric_value)

        return parameters

    def _update_parameters(self, parameters, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            parameters: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.

        Returns:
            模型的参数矩阵.
        """
        num_of_matrix_ops = len(parameters) // 2

        for i in range(1, num_of_matrix_ops + 1):
            parameters['w' + str(i)] -= self.learning_rate * grad['dw' + str(i)]
            parameters['b' + str(i)] -= self.learning_rate * grad['db' + str(i)]

        return parameters

    @staticmethod
    def forward(x, parameters):
        """优化器前向传播.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            parameters: numpy.ndarray, 模型的参数矩阵.

        Returns:
            预测的标签(概率形式)和参数矩阵缓存.
        """
        num_of_matrix_ops = len(parameters) // 2
        caches = []

        a = x  # 除了第一个输入层以外, 其他的是经过激活后的

        for i in range(1, num_of_matrix_ops):
            # 提取参数
            w = parameters['w' + str(i)]
            b = parameters['b' + str(i)]
            x = a  # 提取上一轮的a作为输入

            # 计算输出并激活
            z = np.matmul(x, w.T) + b
            a = relu(z)
            # 保存参数缓存
            cache = (x, w, a)
            caches.append(cache)

        # 输出层的激活函数不同, 单独处理
        w_output = parameters['w' + str(num_of_matrix_ops)]
        b_output = parameters['b' + str(num_of_matrix_ops)]

        z = np.matmul(a, w_output.T) + b_output
        if w_output.shape[0] == 1:
            y_pred = sigmoid(z)
        else:
            y_pred = softmax(z)

        cache = (a, w_output, y_pred)
        caches.append(cache)

        return y_pred, caches

    @staticmethod
    def backward(y_pred, y_true, caches):
        """优化器反向传播.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签(概率形式).
            y_true: numpy.ndarray, 真实的标签.
            caches: numpy.ndarray, 参数缓存.

        Returns:
            优化器的实时梯度矩阵字典.
        """
        num_of_caches = len(caches)
        grad = {}

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # 提取缓存
        (x, w, a) = caches[num_of_caches - 1]
        if y_true.shape[1] == 1:
            # 在二分类的情况下, y_true的标签为零的时候会导致无法计算梯度
            # 这里为了解决这个问题进行了归一化, (-无穷, 0)和(0, 无穷),
            # 实际上到不了无穷, 在10以内
            y_true = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
            da = sigmoid.diff(y_pred, a, y_true)
        else:
            da = softmax.diff(y_true, a)

        # 计算梯度并保存
        dw, db, da_ = StochasticGradientDescent._get_grad(da, y_true, caches[num_of_caches - 1])
        grad['dw' + str(num_of_caches)] = dw
        grad['db' + str(num_of_caches)] = db
        grad['da' + str(num_of_caches)] = da_

        for cache in range(num_of_caches - 1, 0, -1):
            (x, w, a) = caches[cache - 1]

            output = grad.get('da' + str(cache + 1))
            da = relu.diff(output, a)

            # 计算梯度并保存
            dw, db, da_ = StochasticGradientDescent._get_grad(da, y_true, caches[cache - 1])
            grad['dw' + str(cache)] = dw
            grad['db' + str(cache)] = db
            grad['da' + str(cache)] = da_

        return grad

    @staticmethod
    def _get_grad(da, y_true, cache):
        """获取梯度.

        Arguments:
            da: numpy.ndarray, 输出张量的梯度.
            y_true: numpy.ndarray, 真实的标签.
            cache: numpy.ndarray, 参数缓存.

        Returns:
            参数矩阵, 偏置矩阵, 输出张量的梯度.
        """

        # 提取缓存
        (x, w, a) = cache

        dw = np.matmul(da.T, x) / y_true.shape[0]
        db = np.sum(da, axis=0, keepdims=True) / y_true.shape[0]
        da_ = np.matmul(da, w)

        return dw, db, da_


class Adam(StochasticGradientDescent):
    """自适应矩估计优化器.

    References:
        - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
            参照算法1具体实现.

    Notes:
        - 如果想固定随机种子实现, 实现复现的话,
          请在模型实例化的时候将随机种子置为一个常整数.
        - 并采用随机梯度下降作为基础优化算法实现的.
        - 超参数epsilon按照其他的机器学习框架设置的1e-7,
          非原论文的1e-8.
    """

    def __init__(self,
                 name='adam',
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
            learning_rate: float, default=1e-3,
                优化器的学习率.
            beta_1: float, default=0.9,
                一阶矩估计衰减率.
            beta_2: float, default=0.999
                二阶矩估计衰减率.
            epsilon: float, default=1e-7,
                数值稳定的小常数.
        """
        super(Adam, self).__init__(name=name)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def run(self, x, y, epochs, parameters, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            parameters: numpy.ndarray, 模型的参数矩阵.
            args:
                verbose: bool, 显示日志信息.
                loss: str, classicML.losses.Loss 实例,
                    模型使用的损失函数.
                metric: str, classicML.metrics.Metric 实例,
                    模型使用的评估函数.
                callbacks: list,
                    模型训练过程的中间数据记录器.

        Returns:
            模型的参数矩阵.
        """
        verbose, loss, metric, callbacks = _get_optimizer_parameters(args, kwargs)
        num_of_features = x.shape[0]

        # 对Adam进行初始化
        m, v = self._initializer(parameters)

        if verbose is True:
            # 实例化进度条
            self._progressbar = ProgressBar(epochs, loss, metric)

        for epoch in range(1, epochs + 1):
            current = time()
            # 随机选取样本
            random_index = np.random.randint(0, num_of_features)
            # 前向传播
            y_pred, caches = self.forward(x[[random_index], :], parameters)
            # 反向传播
            # Notes: 此时y_true是个scalar
            grad = self.backward(y_pred, np.asarray([y[random_index]]), caches)
            # 更新参数
            parameters, m, v = self._update_parameters(parameters, grad, m, v, epoch)
            # 以下的操作是为了更好可视化的, 不参与实际优化
            if self._progressbar or callbacks is not None:
                # 由于每次只更新一个样本导致预测值的形状和标签不一致,
                # 因此需要多做一次前向传播
                y_pred, _ = self.forward(x, parameters)

                loss_value = loss(y_pred, y)
                metric_value = metric(y_pred, y)

                # 进度条显示
                if self._progressbar:
                    self._progressbar(epoch, current, loss_value, metric_value)
                # 记录callbacks
                if callbacks:
                    _record_callbacks(callbacks, loss_value, metric_value)

        return parameters

    def _update_parameters(self, parameters, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            parameters: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.
            args:
                moment_vector_1: dict of numpy.ndarray,
                    一阶向量.
                moment_vector_2: dict of numpy.ndarray,
                    二阶向量.
                epoch: int, 当前的训练轮数.

        Returns:
            模型的参数矩阵.
        """
        m, v, epoch = args[0], args[1], args[2]

        num_of_matrix_ops = len(parameters) // 2

        for i in range(1, num_of_matrix_ops + 1):
            # 更新一阶矩估计误差
            m['dw' + str(i)] = self.beta_1 * m['dw' + str(i)] + (1 - self.beta_1) * grad['dw' + str(i)]
            m['db' + str(i)] = self.beta_1 * m['db' + str(i)] + (1 - self.beta_1) * grad['db' + str(i)]
            # 更新二阶矩估计误差
            v['dw' + str(i)] = self.beta_2 * v['dw' + str(i)] + (1 - self.beta_2) * (grad['dw' + str(i)] ** 2)
            v['db' + str(i)] = self.beta_2 * v['db' + str(i)] + (1 - self.beta_2) * (grad['db' + str(i)] ** 2)
            # 矫正一阶矩估计误差
            m_hat_w = m['dw' + str(i)] / (1 - np.power(self.beta_1, epoch))
            m_hat_b = m['db' + str(i)] / (1 - np.power(self.beta_1, epoch))
            # 矫正二阶矩估计误差
            v_hat_w = v['dw' + str(i)] / (1 - np.power(self.beta_2, epoch))
            v_hat_b = v['db' + str(i)] / (1 - np.power(self.beta_2, epoch))
            # 更新参数
            parameters['w' + str(i)] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            parameters['b' + str(i)] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return parameters, m, v

    @staticmethod
    def _initializer(parameters):
        """初始化一阶和二阶向量, 算法一实现全初始化为零.

        Arguments:
            parameters: numpy.ndarray, 模型的参数矩阵.

        Returns:
            一阶和二阶向量字典.
        """
        num_of_matrix_ops = len(parameters) // 2

        moment_vector_1 = {}
        moment_vector_2 = {}

        for i in range(1, num_of_matrix_ops + 1):
            moment_vector_1['dw' + str(i)] = np.zeros(parameters['w' + str(i)].shape)
            moment_vector_1['db' + str(i)] = np.zeros(parameters['b' + str(i)].shape)

            moment_vector_2['dw' + str(i)] = np.zeros(parameters['w' + str(i)].shape)
            moment_vector_2['db' + str(i)] = np.zeros(parameters['b' + str(i)].shape)

        return moment_vector_1, moment_vector_2


class GradientDescent(Optimizer):
    """梯度下降优化器.
    """
    def __init__(self, name='gradient_descent', learning_rate=1e-2):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
            learning_rate: float, default=1e-2,
                优化器的学习率.
        """
        super(GradientDescent, self).__init__(name=name)
        self.learning_rate = learning_rate

    def run(self, x, y, epochs, beta, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            beta: numpy.ndarray, 模型的参数矩阵.
            args:
                verbose: bool, 显示日志信息.
                loss: str, classicML.losses.Loss 实例,
                    模型使用的损失函数.
                metric: str, classicML.metrics.Metric 实例,
                    模型使用的评估函数.
        Returns:
            模型的参数矩阵.
        """
        verbose, loss, metric, callbacks = _get_optimizer_parameters(args, kwargs)
        if verbose is True:
            # 实例化进度条
            self._progressbar = ProgressBar(epochs, loss, metric)

        for epoch in range(1, epochs+1):
            current = time()
            # 前向传播
            y_pred, x_hat = self.forward(x, beta)
            # 反向传播
            grad = self.backward(y_pred, y, x_hat)
            # 更新参数
            beta = self._update_parameters(beta, grad)
            # 以下的操作是为了更好可视化的, 不参与实际优化
            if self._progressbar or callbacks is not None:
                if loss.name == 'log_likelihood':
                    loss_value = loss(y, beta, x_hat)
                else:
                    loss_value = loss(y_pred, y)
                metric_value = metric(y_pred, y)

                # 记录callbacks
                if callbacks:
                    _record_callbacks(callbacks, loss_value, metric_value)
                # 进度条显示
                if self._progressbar:
                    self._progressbar(epoch, current, loss_value, metric_value)

        return beta

    def _update_parameters(self, beta, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            beta: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.
        """
        beta -= self.learning_rate * grad

        return beta

    @staticmethod
    def forward(x, parameters):
        """优化器前向传播.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            parameters: numpy.ndarray, 模型的参数矩阵.

        Returns:
            预测的标签(概率形式)和参数(x;1)矩阵.
        """
        number_of_sample = x.shape[0]
        parameters = parameters.reshape(-1, 1)

        x_hat = np.c_[x, np.ones((number_of_sample, 1))]
        cache = np.matmul(x_hat, parameters)
        y_pred = sigmoid(cache)

        return y_pred, x_hat

    @staticmethod
    def backward(y_pred, y_true, x_hat):
        """优化器反向传播.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签(概率形式).
            y_true: numpy.ndarray, 真实的标签.
            x_hat: numpy.ndarray, 属性的参数矩阵.

        Returns:
            优化器的实时梯度矩阵.
        """
        y_true = y_true.reshape(-1, 1)
        error = y_true - y_pred
        grad = np.sum((-x_hat * error), axis=0)
        grad = grad.reshape(-1, 1)

        return grad


class NewtonMethod(Optimizer):
    """牛顿法优化器.
    """
    def __init__(self, name='newton_method'):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
        """
        super(NewtonMethod, self).__init__(name=name)

    def run(self, x, y, epochs, beta, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            beta: numpy.ndarray, 模型的参数矩阵.
            args:
                verbose: bool, 显示日志信息.
                loss: str, classicML.losses.Loss 实例,
                    模型使用的损失函数.
                metric: str, classicML.metrics.Metric 实例,
                    模型使用的评估函数.

        Returns:
            模型的参数矩阵.
        """
        verbose, loss, metric, callbacks = _get_optimizer_parameters(args, kwargs)
        if verbose:
            # 实例化进度条
            self._progressbar = ProgressBar(epochs, loss, metric)

        for epoch in range(1, epochs + 1):
            current = time()
            # 前向传播
            y_pred, x_hat = self.forward(x, beta)
            # 反向传播
            grad = self.backward(y_pred, y, x_hat)
            # 求解海森矩阵
            hessian = self._get_hessian_matrix(y_pred, x_hat)
            # 更新参数
            beta = self._update_parameters(beta, grad, hessian)
            # 以下的操作是为了更好可视化的, 不参与实际优化
            if self._progressbar or callbacks is not None:
                if loss.name == 'log_likelihood':
                    loss_value = loss(y, beta, x_hat)
                else:
                    loss_value = loss(y_pred, y)
                metric_value = metric(y_pred, y)

                # 进度条显示
                if self._progressbar:
                    self._progressbar(epoch, current, loss_value, metric_value)
                # 记录callbacks
                if callbacks:
                    _record_callbacks(callbacks, loss_value, metric_value)

        return beta

    def _update_parameters(self, beta, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            beta: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.
            args:
                hessian: numpy.ndarray, 海森矩阵.
        """
        hessian = args[0]
        beta -= np.matmul(np.linalg.inv(hessian), grad)

        return beta

    @staticmethod
    def forward(x, parameters):
        """优化器前向传播.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            parameters: numpy.ndarray, 模型的参数矩阵.

        Returns:
            预测的标签(概率形式)和参数(x;1)矩阵.
        """
        number_of_sample = x.shape[0]
        parameters = parameters.reshape(-1, 1)

        x_hat = np.c_[x, np.ones((number_of_sample, 1))]
        cache = np.matmul(x_hat, parameters)
        y_pred = sigmoid(cache)

        return y_pred, x_hat

    @staticmethod
    def backward(y_pred, y_true, x_hat):
        """优化器反向传播.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签(概率形式).
            y_true: numpy.ndarray, 真实的标签.
            x_hat: numpy.ndarray, 属性的参数矩阵.

        Returns:
            优化器的实时梯度矩阵.
        """
        y_true = y_true.reshape(-1, 1)
        error = y_true - y_pred
        grad = np.sum((-x_hat * error), axis=0)
        grad = grad.reshape(-1, 1)

        return grad

    @staticmethod
    def _get_hessian_matrix(y_pred, x_hat):
        """计算海森矩阵(二阶导数).

        Arguments:
            y_pred: numpy.ndarray, 预测的标签(概率形式).
            x_hat: numpy.ndarray, 属性的参数矩阵.

        Returns:
            海森矩阵.
        """
        P = np.eye(x_hat.shape[0]) * y_pred * (1 - y_pred)
        H = np.matmul(np.matmul(x_hat.T, P), x_hat)

        return H


class RadialBasisFunctionOptimizer(Optimizer):
    """径向基函数优化器.
    """
    def __init__(self, name='rbf', learning_rate=1e-2):
        """
        Arguments:
            name: str, default=None,
                优化器的名称.
            learning_rate: float, default=1e-2,
                优化器的学习率.
        """
        super(RadialBasisFunctionOptimizer, self).__init__(name=name)
        self.learning_rate = learning_rate

    def run(self, x, y, epochs, parameters, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            parameters: numpy.ndarray, 模型的参数矩阵.
            args:
                verbose: bool, 显示日志信息.
                loss: str, classicML.losses.Loss 实例,
                    模型使用的损失函数.
                metric: str, classicML.metrics.Metric 实例,
                    模型使用的评估函数.
                callbacks: list,
                    模型训练过程的中间数据记录器.
        Returns:
            模型的参数矩阵.
        """
        verbose, loss, metric, callbacks = _get_optimizer_parameters(args, kwargs)

        if verbose is True:
            # 实例化进度条
            self._progressbar = ProgressBar(epochs, loss, metric)

        for epoch in range(1, epochs+1):
            current = time()
            # 前向传播
            y_pred, cache = self.forward(x, parameters)
            # 反向传播
            grad = self.backward(y_pred, y, cache)
            # 更新参数
            parameters = self._update_parameters(parameters, grad)
            # 以下的操作是为了更好可视化的, 不参与实际优化
            if self._progressbar or callbacks is not None:
                loss_value = loss(y_pred, y)
                metric_value = metric(y_pred, y)
                # 进度条显示
                if self._progressbar:
                    self._progressbar(epoch, current, loss_value, metric_value)
                # 记录callbacks
                if callbacks:
                    _record_callbacks(callbacks, loss_value, metric_value)

        return parameters

    def _update_parameters(self, parameters, grad, *args, **kwargs):
        """更新模型的参数.

        Arguments:
            parameters: numpy.ndarray, 模型的参数矩阵.
            grad: numpy.ndarray, 优化器的实时梯度矩阵.

        Returns:
            模型的参数矩阵.
        """
        parameters['w'] -= self.learning_rate * grad['dw']
        parameters['b'] -= self.learning_rate * grad['db']
        parameters['beta'] -= self.learning_rate * grad['dbeta']

        return parameters

    @staticmethod
    def forward(x, parameters):
        """优化器前向传播.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            parameters: numpy.ndarray, 模型的参数矩阵.

        Returns:
            预测的标签(概率形式)和参数矩阵缓存.
        """
        number_of_sample = x.shape[0]
        # 提取参数
        w, b, c, beta = parameters['w'], parameters['b'], parameters['c'], parameters['beta']
        # 获取隐含层神经元的数量
        units = c.shape[0]
        # 初始化高斯径向基函数
        rho = np.zeros((number_of_sample, units))
        x_ci = np.zeros((number_of_sample, units))
        for unit in range(units):
            x_ci[:, unit] = np.linalg.norm(x - c[[unit], ], axis=1) ** 2
            rho[:, unit] = np.exp(-beta[0, unit] * x_ci[:, unit])

        y_pred = np.matmul(rho, w.T) + b
        cache = (rho, x_ci, w, beta)

        return y_pred, cache

    @staticmethod
    def backward(y_pred, y_true, cache):
        """优化器反向传播.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签(概率形式).
            y_true: numpy.ndarray, 真实的标签.
            cache: numpy.ndarray, 参数缓存.

        Returns:
            优化器的实时梯度矩阵.
        """
        (rho, x_ci, w, beta) = cache
        grad = {}

        dy = y_pred - y_true  # 这里虽然叫做dy, 但其实是真实值和预测值之间的误差
        dw = np.matmul(dy.T, rho) / y_true.shape[0]
        db = np.sum(dy, axis=0, keepdims=True) / y_true.shape[0]
        drho = np.matmul(y_pred, w)
        dbeta = np.sum(drho * rho * (-x_ci), axis=0, keepdims=True) / y_true.shape[0]

        grad['dw'] = dw
        grad['db'] = db
        grad['dbeta'] = dbeta

        return grad


class SequentialMinimalOptimization(Optimizer):
    """序列最小最优化算法. SMO算法是一种启发式算法,
    即每次优化两个变量, 使之满足KKT条件; 不断迭代, 最后使得全部变量满足KKT条件.
    整个SMO算法包括: 求解两个变量的二次规划问题和选择变量的启发式方法.

    Attributes:
        alphas: numpy.ndarray,
            拉格朗日乘子数组.
        non_bound_alphas: numpy.ndarray,
            非边界拉格朗日乘子.(硬间隔下和非零拉格朗日乘子一样, 软间隔是非零拉格朗日乘子的子集.)
        error_cache: numpy.ndarray,
            KKT条件的违背值缓存.
        non_zero_alphas: numpy.ndarray,
            非零拉格朗日乘子.
        b: float, default=0,
            偏置项.
        C: float, default=None
            软间隔正则化系数.
        kernel: str, classicML.kernel.Kernels 实例, default=None
            分类器使用的核函数.
        tol: float, default=None
            停止训练的最大误差值.
        epochs: int, default=None
            最大的训练轮数, 如果是-1则表示需要所有的样本满足条件时,
            才能停止训练, 即没有限制.

    References:
        - [李航, 2012.] 统计学习方法 P124~P131
    """
    def __init__(self, name='SMO'):
        super(SequentialMinimalOptimization, self).__init__(name=name)

        self.alphas = None
        self.non_bound_alphas = None
        self.error_cache = None
        self.non_zero_alphas = None
        self.b = None

        self.C = None
        self.kernel = None
        self.tol = None
        self.epochs = None

    def run(self, x, y, *args, **kwargs):
        """运行优化器优化参数.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            args:
                C: float,
                    软间隔正则化系数.
                kernel: str, classicML.kernels.Kernel 实例,
                    分类器使用的核函数.
                tol: float,
                    停止训练的最大误差值.
                epochs: int,
                    最大的训练轮数, 如果是-1则表示需要所有的样本满足条件时,
                    才能停止训练, 即没有限制.
        Returns:
            分类器的支持向量下标数组, 支持向量数组, 拉格朗日乘子数组, 支持向量对应的标签数组和偏置项.
        """
        self.C, self.kernel, self.tol, self.epochs = args[0], args[1], args[2], args[3]

        epoch = 0
        number_of_sample = x.shape[0]
        entire_flag = True

        # 初始化为numpy.ndarray数组, 在Python下更新后会自动转换数据类型为numpy.ndarray,
        # 但是, CC在第一次输入的时候为int, 这里强制为numpy.ndarray用以解决CC下不能动态类型输入.
        self.b = np.asarray([0])
        # 初始化返回参数, 在未知样本的数量时无法初始化.
        self.alphas = np.zeros((number_of_sample, ))
        self.non_bound_alphas = np.zeros((number_of_sample, ))
        self.error_cache = np.zeros((number_of_sample, ))
        self.non_zero_alphas = np.zeros((number_of_sample, ), dtype=bool)

        while (self.epochs == -1) or (epoch < self.epochs):
            pair_of_alpha_changed = 0
            epoch += 1  # 更新迭代次数, 在代码块结尾更新有可能因为提前返回而计数不准确.

            if entire_flag:  # 第一次必须全部全部遍历一遍, 因为初始值全部为零.
                for sample in range(number_of_sample):
                    pair_of_alpha_changed += self._update_parameters(x, y, sample, number_of_sample)
            else:
                non_bound_index = self.non_bound_alphas.nonzero()[0]
                for sample in non_bound_index:
                    pair_of_alpha_changed += self._update_parameters(x, y, sample, number_of_sample)

            if entire_flag is True:
                # 遍历所有样本还是没有更新, 就退出循环.
                if pair_of_alpha_changed == 0:
                    support = self.non_zero_alphas.nonzero()[0]
                    support_vector = x[support]
                    support_alpha = self.alphas[self.non_zero_alphas]
                    support_y = y[support]

                    return support, support_vector, support_alpha, support_y, self.b
                entire_flag = False
            elif pair_of_alpha_changed == 0:
                entire_flag = True

        support = self.non_zero_alphas.nonzero()[0]
        support_vector = x[support]
        support_alpha = self.alphas[self.non_zero_alphas]
        support_y = y[support]

        return support, support_vector, support_alpha, support_y, self.b

    def _update_parameters(self, x, y, *args, **kwargs):
        """SMO算法的内循环(更新参数的具体实现), 寻找第二个要更新的变量alpha_j, 并进行更新.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            args:
                i: int, 第i个样本.
                number_of_sample: int, 样本的总数.

        Returns:
            更新是否成功的标记.
        """
        i, number_of_sample = args[0], args[1]
        y_i = y[i, :]
        alpha_i = self.alphas[i]

        # 提取违背值, 保存为缓存(可能要实时计算).
        if self.non_bound_alphas[i]:
            error_i = self.error_cache[i]
        else:
            error_i = calculate_error(x, y, i, self.kernel, self.alphas, self.non_zero_alphas, self.b)

        # 更新的变化量的绝对值要大于tol, 且alpha要满足软间隔C的条件限制.
        # TODO(Steve R. Sun, tag:code): 直接使用绝对值会导致异常, 但是这样就能正常运行.
        if ((y_i * error_i < -self.tol) and (alpha_i < self.C)) or ((y_i * error_i > self.tol) and (0 < alpha_i)):
            # 存在非边界拉格朗日乘子.
            if np.sum(self.non_bound_alphas) > 0:
                j, error_j = select_second_alpha(error_i, self.error_cache, self.non_bound_alphas)
                if self._update_alpha(x, y, i, j, error_i, error_j):
                    return 1

            # 试图逐元素强制更新.
            for j in np.random.permutation(number_of_sample):
                error_j = calculate_error(x, y, j, self.kernel, self.alphas, self.non_zero_alphas, self.b)
                if self._update_alpha(x, y, i, j, error_i, error_j):
                    return 1

        return 0

    def _update_alpha(self, x, y, i, j, error_i, error_j):
        """更新拉格朗日乘子.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            i: int, 第i个样本.
            j: int, 第j个样本.
            error_i: float, 第i个样本的违背值.
            error_j: float, 第j个样本的违背值.

        Returns:
            是否更新成功.
        """
        if i == j:
            return False

        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()
        y_i = y[i, 0]
        y_j = y[j, 0]

        if y_i != y_j:
            low = max(0.0, alpha_j_old - alpha_i_old)
            high = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            low = max(0.0, alpha_i_old + alpha_j_old - self.C)
            high = min(self.C, alpha_i_old + alpha_j_old)

        if low == high:  # 二维空间对角线重合.
            return False

        x_i = x[[i], :]
        x_j = x[[j], :]

        kappa_ii = self.kernel(x_i, x_i)
        kappa_ij = self.kernel(x_i, x_j)
        kappa_ji = self.kernel(x_j, x_i)
        kappa_jj = self.kernel(x_j, x_j)

        eta = kappa_ii + kappa_jj - 2 * kappa_ij
        if eta <= 0:  # 2-范数小于零.
            return False

        alpha_j_new = alpha_j_old + y_j * (error_i - error_j) / eta
        alpha_j_new = clip_alpha(alpha_j_new, low, high)

        if np.abs(alpha_j_new - alpha_j_old) < 1e-5:  # 更新幅度过小.
            return False

        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

        # 两个变量优化后要重新计算偏置.
        b_i = (-error_i - y_i * kappa_ii * (alpha_i_new - alpha_i_old)
               - y_j * kappa_ji * (alpha_j_new - alpha_j_old) + self.b)
        b_j = (-error_j - y_i * kappa_ij * (alpha_i_new - alpha_i_old)
               - y_j * kappa_jj * (alpha_j_new - alpha_j_old) + self.b)
        self.b = (b_i + b_j) / 2  # 这里采用的周志华机器学习中的方法, 求解平均值更为鲁棒.

        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new

        # 更新缓存.
        self._update_error_cache(x, y)
        self._update_alpha_cache(alpha_i_new, i)
        self._update_alpha_cache(alpha_j_new, j)

        return True

    def _update_error_cache(self, x, y):
        """更新违背值缓存.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
        """
        for i in self.non_bound_alphas.nonzero()[0]:
            self.error_cache[i] = calculate_error(x, y, i, self.kernel, self.alphas, self.non_zero_alphas, self.b)

    def _update_alpha_cache(self, alpha, index):
        """更新拉格朗日乘子缓存.

        Arguments:
            alpha: float, 拉格朗日乘子.
            index: int, 拉格朗日乘子的下标.
        """
        self.non_zero_alphas[index] = (alpha > 0)
        self.non_bound_alphas[index] = int(0 < alpha < self.C)


# Aliases.
SGD = StochasticGradientDescent
SMO = SequentialMinimalOptimization