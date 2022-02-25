import numpy as np

from classicML import _cml_precision
from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.api.models import DecisionStumpClassifier
from classicML.backend import bootstrap_sampling


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
    """
    def __init__(self):
        """初始化Bagging分类器.
        """
        super(BaggingClassifier, self).__init__()

        self.estimators = []
        self.BaseLearner = None
        self.seed = -1

        self.is_trained = False

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
        if self.is_trained is False:
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
