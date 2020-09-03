import numpy as np

from .initializers import beta_initializer
from .optimziers import GD, GradientDescent, Newton, NewtonMethod, forward


class LogisticRegression:
    """逻辑回归"""
    def __init__(self, seed=None):
        """
        逻辑回归初始化
        Parameters
        ----------
        seed : int, default=None, optional
            随机种子

        Arguments
        ---------
        fitted : bool, default=False
            训练后此参数被自动置为True, 用于标记是否被训练
        """
        np.random.seed(seed)

        self.fitted = False

    def compile(self, loss=None, optimizer='GD', learning_rate=1e-2, metrics=None):
        """
        进行编译, 配置参数项
        Parameters
        ----------
        loss : str or function, default=None
            模型使用的损失函数
        optimizer : {'GD', 'Newton'}
            模型使用的优化器
        learning_rate : float, default=1e-2
            梯度下降优化器的学习率, 使用牛顿法是此参数无效
        metrics : str or function, default=None
            评估函数
        """
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrcis = metrics

    def fit(self, x, y, epochs=1, verbose=True):
        """
        训练
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
        """

        # 避免dtype是object, 从而引发异常
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(int)
        num_of_x, attr_of_x = x.shape

        # 参数初始化
        self.beta = beta_initializer(attr_of_x)

        # 优化器优化
        if self.optimizer in ('GD', 'GradientDescent', GD, GradientDescent):
            self.beta = GradientDescent(x, y, epochs, verbose, self.beta, self.learning_rate, self.loss, self.metrcis)
        elif self.optimizer in ('Newton', 'NewtonMethod', Newton, NewtonMethod):
            self.beta = NewtonMethod(x, y, epochs, verbose, self.beta, self.loss, self.metrcis)
        else:
            raise Exception('请输入正确的优化器')

        self.fitted = True

        return self

    def predict(self, x):
        """
        预测
        Parameters
        ----------
        x : numpy.ndarray or array-like
            特征数据

        Returns
        -------
        y_pred : numpy.ndarray
            逻辑回归预测的概率数组
        """
        if self.fitted is False:
            raise Exception('你必须先进行训练')
        y_pred, _ = forward(x, self.beta)

        return y_pred