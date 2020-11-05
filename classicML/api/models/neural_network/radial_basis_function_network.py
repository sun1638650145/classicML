import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.backend import get_initializer
from classicML.backend import get_optimizer
from classicML.backend import get_loss
from classicML.backend import get_metric


class RadialBasisFunctionNetwork(object):
    """径向基函数网络.

    Attributes:
        seed: int, default=None,
            随机种子.
        initializer: str, 径向基函数网络的初始化器.
        hidden_units: int, 径向基函数网络的隐含层神经元数量.
        optimizer: str, classicML.optimizers.Optimizer 实例, 径向基函数网络的优化器.
        loss: str, classicML.losses.Loss 实例, default='mse'
            径向基函数网络使用的损失函数.
        metric: str, classicML.metrics.Metric 实例, default='accuracy'
            径向基函数网络使用的评估函数.
        parameters: numpy.ndarray,
            径向基函数网络的参数矩阵.
        is_trained: bool, default=None,
            径向基函数网络训练后将被标记为True.
    """
    def __init__(self, seed=None):
        """初始化径向基函数网络.

        Arguments:
            seed: int, default=None,
                随机种子.
        """
        super(RadialBasisFunctionNetwork, self).__init__()
        self.seed = seed

        self.initializer = None
        self.hidden_units = None
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.parameters = None
        self.is_trained = False

    def compile(self, hidden_units, optimizer='rbf', loss='mse', metric='accuracy'):
        """编译径向基函数网络, 配置训练时使用的超参数.

        Arguments:
            hidden_units: int, 径向基函数网络的隐含层神经元数量.
            optimizer: str, classicML.optimizers.Optimizer 实例,
                径向基函数网络使用的优化器.
            loss: str, classicML.losses.Loss 实例, default='mse'
                径向基函数网络使用的损失函数.
            metric: str, classicML.metrics.Metric 实例, default='accuracy'
                径向基函数网络使用的评估函数.

        Notes:
            - 注意RBF只能是用RadialBasisFunctionOptimizer优化器,
              之所以开放优化器选项, 只是为了满足用户修改学习率的需求.
            - 使用交叉熵作为损失函数有潜在异常的可能性,
              除隐含层神经元个数和学习率之外, 建议使用默认参数.
        """
        self.hidden_units = hidden_units

        self.initializer = get_initializer('rbf_normal', self.seed)
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)
        self.metric = get_metric(metric)

    def fit(self, x, y, epochs=1, verbose=True, callbacks=None):
        """训练径向基函数网络.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            verbose: bool, default=True, 可选
                显示日志信息.
            callbacks: list, default=None,
                模型训练过程的中间数据记录器.
        Returns:
            RadialBasisFunctionNetwork实例.
        """
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(int)

        # 使用初始化器初始化参数
        self.parameters = self.initializer(hidden_units=self.hidden_units)
        # 使用优化器优化
        self.parameters = self.optimizer(x, y, epochs, self.parameters, verbose, self.loss, self.metric, callbacks)
        # 标记训练完成
        self.is_trained = True

        return self

    def predict(self, x):
        """使用径向基函数网络进行预测.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.

        Returns:
            径向基函数网络预测的概率.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        y_pred, _ = self.optimizer.forward(x, self.parameters)

        return y_pred