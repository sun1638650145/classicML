from .initializers import *
from .optimizers import *
from .history import *


class RadialBasisFuncionNetwork:
    """生成一个径向基函数网络"""

    def __init__(self, seed=None):
        """进行初始化"""
        self.seed = seed

    def compile(self, learning_rate, hidden_units):
        """编译超参数"""
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units

    def fit(self, x, y, epochs=1, verbose=False):
        """训练RBF网络"""

        # 参数初始化
        epochs = int(epochs)  # 使用科学记数法的时候也能正常
        self.parameters = rbf_initializer(self.hidden_units, self.seed)

        # history初始化
        self.history = History(epochs)

        # 模型优化
        parameters, loss, acc = RBFOptimizer(x, y, epochs, verbose, self.parameters, self.learning_rate)

        self.parameters = parameters
        self.history.add_loss(loss)
        self.history.add_accuracy(acc)

        return self.history

    def predict(self, x):
        """进行预测"""
        if not hasattr(self, 'parameters'):
            raise Exception('你必须先进行训练')

        y_pred, _ = rbf_forward(x, self.parameters)
        if y_pred.shape[1] == 1:
            ans = np.zeros(y_pred.shape)
            ans[y_pred >= 0.5] = 1
        else:
            ans = np.argmax(y_pred, axis=1)

        return ans