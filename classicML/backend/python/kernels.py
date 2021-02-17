"""classicML的核函数."""
import numpy as np


class Kernel(object):
    """核函数的基类.

    Attributes:
        name: str, default='kernel',
            核函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
    """
    def __init__(self, name='kernel'):
        """
        Arguments:
            name: str, default='kernel',
                核函数名称.
        """
        self.name = name

    def __call__(self, x_i, x_j):
        raise NotImplementedError


class Linear(Kernel):
    """线性核函数.
    """
    def __init__(self, name='linear'):
        super(Linear, self).__init__(name=name)

    def __call__(self, x_i, x_j):
        """函数实现.

        Arguments:
            x_i: numpy.ndarray, 第一组特征向量.
            x_j: numpy.ndarray, 第二组特征向量.

        Returns:
            核函数映射后的特征向量.
        """
        kappa = np.matmul(x_j, x_i.T)

        return np.asmatrix(kappa)


class Polynomial(Kernel):
    """多项式核函数.

    Attributes:
        name: str, default='poly',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
        degree: int, default=3,
            多项式的次数.
    """
    def __init__(self, name='poly', gamma=1.0, degree=3):
        super(Polynomial, self).__init__(name=name)

        self.gamma = gamma
        self.degree = degree

    def __call__(self, x_i, x_j):
        """函数实现.

        Arguments:
            x_i: numpy.ndarray, 第一组特征向量.
            x_j: numpy.ndarray, 第二组特征向量.

        Returns:
            核函数映射后的特征向量.
        """
        kappa = self.gamma * np.power(np.matmul(x_j, x_i.T), self.degree)

        return np.asmatrix(kappa)


class RBF(Kernel):
    """径向基核函数.

    Attributes:
        name: str, default='rbf',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
    """
    def __init__(self, name='rbf', gamma=1.0):
        super(RBF, self).__init__(name=name)

        self.gamma = gamma

    def __call__(self, x_i, x_j):
        """函数实现.

        Arguments:
            x_i: numpy.ndarray, 第一组特征向量.
            x_j: numpy.ndarray, 第二组特征向量.

        Returns:
            核函数映射后的特征向量.
        """
        kappa = np.exp(self.gamma * -np.sum(np.power(x_j - x_i, 2), axis=1))

        return np.asmatrix(kappa)


class Gaussian(RBF):
    """高斯核函数.
        具体实现参看径向基核函数.
    """
    def __init__(self, name='gaussian', gamma=1.0):
        super(Gaussian, self).__init__(name=name, gamma=gamma)


class Sigmoid(Kernel):
    """Sigmoid核函数.

    Attributes:
        name: str, default='sigmoid',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
        beta: float, default=1.0,
            核函数参数.
        theta: float, default=-1.0,
            核函数参数.
    """
    def __init__(self, name='sigmoid', gamma=1.0, beta=1.0, theta=-1.0):
        super(Sigmoid, self).__init__(name=name)

        self.gamma = gamma
        self.beta = beta
        self.theta = theta

    def __call__(self, x_i, x_j):
        """函数实现.

        Arguments:
            x_i: numpy.ndarray, 第一组特征向量.
            x_j: numpy.ndarray, 第二组特征向量.

        Returns:
            核函数映射后的特征向量.
        """
        kappa = self.gamma * np.tanh(self.beta * np.matmul(x_j, x_i.T) + self.theta)

        return np.asmatrix(kappa)
