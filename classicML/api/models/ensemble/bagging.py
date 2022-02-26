from pickle import dumps, loads

import numpy as np

from classicML import _cml_precision
from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.api.models import DecisionStumpClassifier
from classicML.backend import bootstrap_sampling
from classicML.backend import io


class BaggingClassifier(BaseModel):
    """Bagging分类器.

    Attributes:
        estimators: list of `BaseLearner`实例,
            Bagging集成的基学习器列表.
        BaseLearner: `BaseLearner`对象, default=None
            Bagging使用的基学习器.
        seed: int, default=随机整数,
            初始自助采样的随机种子.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self):
        """初始化Bagging分类器.
        """
        super(BaggingClassifier, self).__init__()

        self.estimators = []
        self.BaseLearner = None
        self.seed = -1

        self.is_trained = False
        self.is_loaded = False

    def compile(self, base_algorithm=DecisionStumpClassifier, seed=np.random.randint(65535)):
        """编译Bagging分类器.

        Args:
            base_algorithm: `BaseLearner`对象, default=DecisionStumpClassifier,
                Bagging使用的基学习器算法;
                目前只实现了`DecisionStumpClassifier`, 未来将接入更多算法.
            seed: int, default=随机整数,
                初始自助采样的随机种子.
        """
        self.BaseLearner = base_algorithm
        self.seed = seed

    def fit(self, x, y, n_estimators=10):
        """训练Bagging分类器.

        Args:
            x: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            n_estimators: int, default=10,
                基学习器集成的数量.

        Return:
            BaggingClassifier实例.
        """
        x = np.asarray(x, dtype=_cml_precision.float)
        y = np.asarray(y, dtype=_cml_precision.int)

        for _ in range(n_estimators):
            # 进行自助采样.
            x_, y_ = bootstrap_sampling(x, y, self.seed)
            self.seed += 1  # 对随机种子进行更新.
            # 基于自助采样产生的样本分布训练基学习器.
            base_learner = self.BaseLearner().fit(x_, y_)
            self.estimators.append(base_learner)

        # 标记训练完成.
        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """使用Bagging分类器进行预测.

        Args:
            x: numpy.ndarray or array-like,
                特征数据.

        Return:
            BaggingClassifier预测的结果.

        Raise:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        y_pred = np.zeros(shape=(x.shape[0]), dtype=_cml_precision.float)

        for estimator in self.estimators:
            y_pred += estimator.predict(x)

        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1

        return y_pred.astype(_cml_precision.int)

    def score(self, x, y):
        """在预测模式下计算准确率.

        Arguments:
            x: array-like, 特征数据.
            y: array-like, 标签.

        Returns:
            当前的准确率.
        """
        return super(BaggingClassifier, self).score(x, y)

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            KeyError: 模型权重加载失败.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='r',
                                                   model_name='BaggingClassifier')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.BaseLearner = loads(compile_ds.attrs['base_algorithm'].tobytes())

            self.estimators = loads(weights_ds.attrs['estimators'].tobytes())
            self.seed = weights_ds.attrs['seed']
            # 标记加载完成.
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

        References:
            - [如何存储原始的二进制数据](https://docs.h5py.org/en/2.3/strings.html)
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='w',
                                                   model_name='BaggingClassifier')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['base_algorithm'] = np.void(dumps(self.BaseLearner))

            weights_ds.attrs['estimators'] = np.void(dumps(self.estimators))
            weights_ds.attrs['seed'] = self.seed
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')
