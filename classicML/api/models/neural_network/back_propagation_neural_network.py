import copy

import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import get_initializer
from classicML.backend import get_optimizer
from classicML.backend import get_loss
from classicML.backend import get_metric
from classicML.backend import io


class BackPropagationNeuralNetwork(BaseModel):
    """BP神经网络.

    Attributes:
        seed: int, default=None,
            随机种子.
        initializer: str or classicML.initializers.Initializer 实例, default='random_normal'
            初始化器.
        network_structure: list,
            神经网络的结构, 定义神经网络的隐含层和输出层的神经元个数(输入层目前将自动推理)
        optimizer: str, classicML.optimizers.Optimizer 实例, default='sgd'
            神经网络使用的优化器.
        loss: str, classicML.losses.Loss 实例, default='crossentropy'
            神经网络使用的损失函数.
        metric: str, classicML.metrics.Metric 实例, default='accuracy'
            神经网络使用的评估函数.
        parameters: dict of numpy.ndarray,
            神经网络的参数矩阵.
        is_trained: bool, default=None,
            神经网络训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self, seed=None, initializer=None):
        """初始化神经网络.

        Arguments:
            seed: int, default=None,
                随机种子.
            initializer: str or classicML.initializers.Initializer 实例, default='he_normal'
                初始化器.
        """
        super(BackPropagationNeuralNetwork, self).__init__()
        self.seed = seed
        self.initializer = initializer

        self.network_structure = None
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.parameters = dict()
        self.is_trained = False
        self.is_loaded = False

    def compile(self, network_structure, optimizer='sgd', loss='crossentropy', metric='accuracy'):
        """编译神经网络, 配置训练时使用的超参数.

        Arguments:
            network_structure: list,
                神经网络的结构, 定义神经网络的隐含层和输出层的神经元个数(输入层目前将自动推理);
                例如: [3, 1]是一个隐含层3个神经元和输出层1个神经元的网络,
                     [5, 5, 2]是一个有两个隐含层每层有5个神经元和输出层2个神经元的网络,
            optimizer: str, classicML.optimizers.Optimizer 实例, default='sgd'
                神经网络使用的优化器.
            loss: str, classicML.losses.Loss 实例, default='crossentropy'
                神经网络使用的损失函数.
            metric: str, classicML.metrics.Metric 实例, default='accuracy'
                神经网络使用的评估函数.
        """
        self.network_structure = network_structure

        self.initializer = get_initializer(self.initializer, self.seed)
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)
        self.metric = get_metric(metric)

    def fit(self, x, y, epochs=1, verbose=True, callbacks=None):
        """训练神经网络.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1, 训练的轮数.
            verbose: bool, default=True, 可选
                显示日志信息.
            callbacks: list, default=None,
                模型训练过程的中间数据记录器.
        Returns:
            BackPropagationNeuralNetwork实例.
        """
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(int)

        # 没有使用权重文件, 则初始化模型结构和使用初始化器初始化参数.
        if self.is_loaded is False:
            _attributes_of_feature = x.shape[1]
            network_structure = copy.deepcopy(self.network_structure)
            network_structure.insert(0, _attributes_of_feature)  # 插入输入层结构

            self.parameters = self.initializer(attributes_or_structure=network_structure)
        # 使用优化器优化
        self.parameters = self.optimizer(x, y, epochs, self.parameters, verbose, self.loss, self.metric, callbacks)
        # 标记训练完成
        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """使用神经网络进行预测.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.

        Returns:
            神经网络预测的概率.

        Raises:
            ValueError: 模型没有训练的错误.
            TypeError: 输入参数的类型错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')
        if not isinstance(x, np.ndarray):
            CLASSICML_LOGGER.error('请检查参数的数据类型')
            raise TypeError('参数的类型错误')

        y_pred, _ = self.optimizer.forward(x, self.parameters)

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
                                                   model_name='BackPropagationNeuralNetwork')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.network_structure = compile_ds.attrs['network_structure']
            self.optimizer = get_optimizer(compile_ds.attrs['optimizer'])
            self.loss = get_loss(compile_ds.attrs['loss'])
            self.metric = get_metric(compile_ds.attrs['metric'])
            for attr in weights_ds.attrs:
                self.parameters[attr] = weights_ds.attrs[attr]
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
                                                   model_name='BackPropagationNeuralNetwork')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['network_structure'] = self.network_structure
            compile_ds.attrs['optimizer'] = self.optimizer.name
            compile_ds.attrs['loss'] = self.loss.name
            compile_ds.attrs['metric'] = self.metric.name
            for parameter in self.parameters:
                weights_ds.attrs[parameter] = self.parameters[parameter]
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')
