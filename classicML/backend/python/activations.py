"""classicML的激活函数."""
import numpy as np


class Activation(object):
    """激活函数基类.

    Attributes:
        name: str, default='activation',
            激活函数名称.

    Raises:
       NotImplementedError: __call__, diff方法需要用户实现.
    """
    def __init__(self, name='activation'):
        self.name = name

    def __call__(self, z):
        """函数实现.

        Arguments:
            z: numpy.ndarray, 输入张量.
        """
        raise NotImplementedError

    def diff(self, output, a, *args, **kwargs):
        """函数的导数(微分).

        Arguments:
            output: numpy.ndarray, 输出张量.
            a: numpy.ndarray, 输入张量.
        """
        raise NotImplementedError


class Relu(Activation):
    """ReLU激活函数.
    """
    def __init__(self, name='relu'):
        super(Relu, self).__init__(name=name)

    def __call__(self, z):
        """
        Arguments:
            z: numpy.ndarray, 输入的张量.

        Returns:
            经过激活后的张量.
        """
        result = np.maximum(0, z)

        return result

    def diff(self, output, a, *args, **kwargs):
        """ReLU函数的微分.

        Arguments:
            output: numpy.ndarray, 前向传播输出的张量.
            a: numpy.ndarray, 输入的张量.

        Notes:
            ReLU函数在大于零区间的导数应该是恒为一, 如果按此计算在实际应用上会随着训练轮数的增加, 最后模型的输出是一个随机概率,
            作者个人认为原因是随着轮数的增加, 大部分神经元都恒为激活态, 成为一个线性操作(缩放实际上相当于不参与计算了).
            在实际应用中使用原值发现可以避免这种想象.
        """
        da = np.asarray(output)
        da[a <= 0] = 0

        return da


class Sigmoid(Activation):
    """Sigmoid激活函数.
    """
    def __init__(self, name='sigmoid'):
        super(Sigmoid, self).__init__(name=name)

    def __call__(self, z):
        """
        Arguments:
            z: numpy.ndarray, 输入的张量.

        Returns:
            经过激活后的张量.
        """
        result = 1 / (1 + np.exp(-z))

        return result

    def diff(self, output, a, *args, **kwargs):
        """Sigmoid的导数(微分).

        Arguments:
            output: numpy.ndarray, 输出张量.
            a: numpy.ndarray, 输入张量.
            args:
                y_true: numpy.ndarray, 真实的标签.

        Returns:
            Sigmoid的导数(微分).

        Notes:
            Sigmoid的导数f' = a * (1 - a),
            但是作为输出层就需要乘上误差.
        """
        y_true = args[0]
        error = y_true - output

        da = a * (1 - a) * error

        return da


class Softmax(Activation):
    """Softmax激活函数.
    """
    def __init__(self, name='softmax'):
        super(Softmax, self).__init__(name=name)

    def __call__(self, z):
        """
        Arguments:
            z: numpy.ndarray, 输入的张量.

        Returns:
            经过激活后的张量.
        """
        z -= np.max(z)  # 为了避免溢出减去最大值
        result = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        return result

    def diff(self, output, a, *args, **kwargs):
        """Softmax函数的微分.

        Arguments:
            output: numpy.ndarray, 前向传播输出的张量.
            a: numpy.ndarray, 输入的张量.

        References:
            https://blog.csdn.net/qq_38032064/article/details/90599547?
        """
        da = a - output

        return da


# Instances.
relu = Relu()
sigmoid = Sigmoid()
softmax = Softmax()
