import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import get_initializer
from classicML.backend import get_optimizer
from classicML.backend import get_loss
from classicML.backend import get_metric
from classicML.backend import io


class LogisticRegression(BaseModel):
    """逻辑回归.

    Attributes:
        seed: int, default=None,
            随机种子.
        initializer: str or classicML.initializers.Initializer 实例, default='random_normal'
            初始化器.
        optimizer: str, classicML.optimizers.Optimizer 实例, default='newton'
            模型使用的优化器.
        loss: str, classicML.losses.Loss 实例, default='log_likelihood'
            模型使用的损失函数.
        metric: str, classicML.metrics.Metric 实例, default='accuracy'
            模型使用的评估函数.
        beta: numpy.ndarray,
            模型的参数矩阵.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self, seed=None, initializer=None):
        """初始化模型.

        Arguments:
             seed: int, default=None,
                随机种子.
             initializer: str or classicML.initializers.Initializer 实例, default='random_normal'
                初始化器.
        """
        super(LogisticRegression, self).__init__()
        self.seed = seed
        self.initializer = initializer

        self.initializer = None
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.beta = None
        self.is_trained = False
        self.is_loaded = False

    def compile(self, optimizer='newton', loss='log_likelihood', metric='accuracy'):
        """编译模型, 配置训练时使用的超参数.

        Arguments:
            optimizer: str, classicML.optimizers.Optimizer 实例, default='newton'
                模型使用的优化器.
            loss: str, classicML.losses.Loss 实例, default='log_likelihood'
                模型使用的损失函数.
            metric: str, classicML.metrics.Metric 实例, default='accuracy'
                模型使用的评估函数.
        """
        self.initializer = get_initializer(self.initializer, self.seed)
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)
        self.metric = get_metric(metric)

    def fit(self, x, y, epochs=1, verbose=True, callbacks=None):
        """训练模型.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            verbose: bool, default=True, 可选
                显示日志信息.
            callbacks: list, default=None,
                模型训练过程的中间数据记录器.
        Returns:
            LogisticRegression实例.
        """
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(int)

        _attributes_of_feature = x.shape[1]

        # 没有使用权重文件, 则使用初始化器初始化参数
        if self.is_loaded is False:
            self.beta = self.initializer(attributes_or_structure=_attributes_of_feature)
        # 使用优化器优化
        self.beta = self.optimizer(x, y, epochs, self.beta, verbose, self.loss, self.metric, callbacks)
        # 标记训练完成
        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """模型进行预测.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.

        Returns:
            LogisticRegression预测的概率.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        y_pred, _ = self.optimizer.forward(x, self.beta)
        y_pred = np.squeeze(y_pred)

        return y_pred

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            KeyError: 模型权重加载失败.

        Notes:
            模型将不会加载关于优化器的超参数.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='r',
                                                   model_name='LogisticRegression')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.optimizer = get_optimizer(compile_ds.attrs['optimizer'])
            self.loss = get_loss(compile_ds.attrs['loss'])
            self.metric = get_metric(compile_ds.attrs['metric'])
            self.beta = weights_ds.attrs['beta']
            # 标记加载完成
            self.is_loaded = True
        except KeyError:
            CLASSICML_LOGGER.error('模型权重加载失败, 请检查文件是否损坏')
            raise KeyError('模型权重加载失败')

    def save_weights(self, filepath):
        """将模型权重保存为一个HDF5文件.

        Arguments:
            filepath: str, 权重文件保存的路径.

        Raises:
            TypeError: 模型权重保存失败.

        Notes:
            模型将不会保存关于优化器的超参数.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='w',
                                                   model_name='LogisticRegression')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['optimizer'] = self.optimizer.name
            compile_ds.attrs['loss'] = self.loss.name
            compile_ds.attrs['metric'] = self.metric.name
            weights_ds.attrs['beta'] = self.beta
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')
