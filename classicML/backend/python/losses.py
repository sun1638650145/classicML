"""classicML的损失函数."""
import numpy as np


class Loss(object):
    """损失函数的基类.

    Attributes:
        name: str, default='loss',
            损失函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
    """
    def __init__(self, name='loss'):
        """
        Arguments:
            name: str, default=None,
                损失函数名称.
        """
        self.name = name

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """函数实现."""
        raise NotImplementedError


class BinaryCrossentropy(Loss):
    """二分类交叉熵损失函数.
    """
    def __init__(self, name='binary_crossentropy'):
        super(BinaryCrossentropy, self).__init__(name=name)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的损失值.
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_shape = y_true.shape[0]
        loss = -(np.matmul(y_true.T, np.log(y_pred)) + np.matmul(1 - y_true.T, np.log(1 - y_pred))) / y_shape

        return np.squeeze(loss)


class CategoricalCrossentropy(Loss):
    """多分类交叉熵损失函数.
    """
    def __init__(self, name='categorical_crossentropy'):
        super(CategoricalCrossentropy, self).__init__(name=name)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的损失值.
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_shape = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / y_shape

        return loss


class Crossentropy(Loss):
    """交叉熵损失函数,
    将根据标签的实际形状自动使用二分类或者多分类损失函数.
    """
    def __init__(self, name='crossentropy'):
        super(Crossentropy, self).__init__(name=name)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的损失值.
        """
        if y_pred.shape[1] == 1:
            crossentropy = BinaryCrossentropy()
        else:
            crossentropy = CategoricalCrossentropy()
        loss = crossentropy(y_pred, y_true)

        return loss


class LogLikelihood(Loss):
    """对数似然损失函数.
    """
    def __init__(self, name='log_likelihood'):
        super(LogLikelihood, self).__init__(name=name)

    def __call__(self, y_true, beta, *args, **kwargs):
        """函数实现.
        
        Arguments:
            y_true: numpy.ndarray, 真实的标签.
            beta: numpy.ndarray, 模型的参数矩阵.
            args:
                x_hat: numpy.ndarray, 属性的参数矩阵.

        Returns:
            当前的损失值.
        """
        x_hat = args[0]
        y_true = y_true.reshape(-1, 1)
        beta = beta.reshape(-1, 1)
        loss = -y_true * np.matmul(x_hat, beta) + np.log(1 + np.exp(np.matmul(x_hat, beta)))

        return np.sum(loss)


class MeanSquaredError(Loss):
    """均方误差损失函数.
    """
    def __init__(self, name='mean_squared_error'):
        super(MeanSquaredError, self).__init__(name=name)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的损失值.
        """
        y_shape = y_true.shape[0]
        loss = np.sum(y_pred - y_true) ** 2 / (2 * y_shape)

        return np.squeeze(loss)

# Aliases.
MSE = MeanSquaredError
