import copy
from .initializers import *
from .optimizers import *
from .history import *


class BackPropagationNeuralNetwork:
    """生成一个BP神经网络"""

    def __init__(self, seed=None, initializer='he_normal'):
        """进行初始化"""
        self.seed = seed
        self.initializer = initializer

    def compile(self, layer_dim, loss=None, optimizer='SGD', learning_rate=1e-2, metrics=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.layer_dim = layer_dim
        self.loss_function = loss  # loss为None将自动推理 优先使用cross_entropy
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics
        # 使用Adam需要用的参数，使用GD和SGD时参数无效
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def fit(self, x, y, epochs=1, verbose=True):
        """训练神经网络"""
        num_of_x, attr_of_x = x.shape
        layer_dim = copy.deepcopy(self.layer_dim)
        # 插入输入层
        layer_dim.insert(0, attr_of_x)
        # 对y增维便于计算
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 参数初始化
        epochs = int(epochs)  # 使用科学记数法的时候也能正常
        if self.initializer == 'he_normal':
            self.parameters = he_normal(layer_dim, self.seed)
        elif self.initializer in ('xavier', 'glorot'):
            self.parameters = xavier(layer_dim, self.seed)

        # history初始化
        self.history = History(epochs)

        # 优化器优化
        if self.optimizer in ('GD', 'GradientDescent', GD, GradientDescent):
            parameters, loss, acc = GradientDescent(x, y, epochs, verbose, self.parameters, self.learning_rate, self.loss_function, self.metrics)
        elif self.optimizer in ('SGD', 'StochasticGradientDescent', SGD, StochasticGradientDescent):
            parameters, loss, acc = SGD(x, y, epochs, verbose, self.parameters, self.learning_rate, self.loss_function, self.metrics, self.seed)
        elif self.optimizer in ('Adam', Adam):
            parameters, loss, acc = Adam(x, y, epochs, verbose, self.parameters, self.loss_function, self.metrics, self.seed, self.learning_rate, self.beta_1, self.beta_2, self.epsilon)

        self.parameters = parameters
        self.history.add_loss(loss)
        self.history.add_accuracy(acc)

        return self.history

    def predict(self, x):
        """进行预测"""
        if not hasattr(self, "parameters"):
            raise Exception('你必须先进行训练')

        y_pred, _ = forward(x, self.parameters)
        if y_pred.shape[1] == 1:
            ans = np.zeros(y_pred.shape)
            ans[y_pred >= 0.5] = 1
        else:
            ans = np.argmax(y_pred, axis=1)

        return ans
