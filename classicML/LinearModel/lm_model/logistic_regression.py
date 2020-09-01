import numpy as np
from .initializers import beta_initializer
from .optimziers import GD, GradientDescent, Newton, NewtonMethod, forward


class LogisticRegression:
    """逻辑回归"""
    def __init__(self, seed=None):
        np.random.seed(seed)
        self.fitted = False

    def compile(self, loss=None, optimizer='GD', learning_rate=1e-2, metrics=None):
        """配置"""
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrcis = metrics

    def fit(self, x, y, epochs=1, verbose=True):
        """训练"""
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
            self.beta = NewtonMethod(x, y, epochs, verbose, self.beta, self.learning_rate, self.loss, self.metrcis)
        else:
            raise Exception('请输入正确的优化器')

        self.fitted = True

        print()

        return self

    def predict(self, x):
        """预测"""
        if self.fitted is False:
            raise Exception('你必须先进行训练')
        y_pred, _ = forward(x, self.beta)

        return y_pred