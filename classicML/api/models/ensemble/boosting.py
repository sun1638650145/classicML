import numpy as np

from classicML import _cml_precision
from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.api.models import TwoLevelDecisionTreeClassifier


EPSILON = 1e-36  # 常小数.


class AdaBoostClassifier(BaseModel):
    """AdaBoost分类器.

    Attributes:
        estimators: list of `BaseLearner`实例,
            AdaBoost集成的基学习器列表.
        alpha_list: list of float,
            AdaBoost集成的基学习器对应的权重.
        BaseLearner: `BaseLearner`对象, default=None
            AdaBoost使用的基学习器.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
    """
    def __init__(self):
        """初始化AdaBoost分类器.
        """
        super(AdaBoostClassifier, self).__init__()

        self.estimators = []
        self.alpha_list = []
        self.BaseLearner = None

        self.is_trained = False

    def compile(self, base_algorithm=TwoLevelDecisionTreeClassifier):
        """编译AdaBoost分类器.

        Args:
            base_algorithm: `BaseLearner`对象, default=TwoLevelDecisionTreeClassifier,
                AdaBoost使用的基学习器算法;
                目前只实现了`TwoLevelDecisionTreeClassifier`, 未来将接入更多算法.
        """
        self.BaseLearner = base_algorithm

    def fit(self, x, y, max_estimators=50):
        """训练AdaBoost分类器.

        Args:
            x: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            max_estimators: int, default=50,
                基学习器集成的最大数量(训练的最大轮数).

        Return:
            AdaBoostClassifier实例.
        """
        x = np.asarray(x, dtype=_cml_precision.float)
        y = np.asarray(y, dtype=_cml_precision.int)

        # 样本分布初始化.
        sample_distribution = np.ones(len(x)) / len(x)

        for _ in range(max_estimators):
            # 基于样本分布训练基学习器.
            base_learner = self.BaseLearner().fit(x, y, sample_distribution=sample_distribution)
            y_pred = base_learner.predict(x)
            # 计算当前基学习器的误差(损失).
            error = np.sum(sample_distribution[y_pred != y])
            # 计算基学习器的权重, 保存权重和基学习器.
            alpha = np.log((1 - error) / np.maximum(error, EPSILON)) / 2  # 使用常小数避免除零.
            self.alpha_list.append(alpha)
            self.estimators.append(base_learner)
            # 更新样本分布.
            error_index = np.ones(len(x))
            error_index[y_pred == y] = -1
            sample_distribution *= np.exp(error_index * alpha)
            # 进行规范化(和为1).
            sample_distribution /= np.sum(sample_distribution)
            # 退出条件, 与原始伪码略有不同.
            # 1. 退出条件增加误差过小, 误差过小时不必集成更多学习器, 降低过拟合和运算时间.
            # 2. 退出条件放在最后, 避免出现极端情况第一个基学习器误差大于0.5, AdaBoost为空.
            if error > 0.5 or error < EPSILON:
                break

        # 标记训练完成.
        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """使用AdaBoost分类器进行预测.

        Args:
            x: numpy.ndarray or array-like,
                特征数据.

        Return:
            AdaBoostClassifier预测的结果.

        Raise:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        y_pred = np.zeros(shape=x.shape[0], dtype=_cml_precision.float)

        for i, (estimator, alpha) in enumerate(zip(self.estimators, self.alpha_list)):
            y_pred += alpha * estimator.predict(x)

        y_pred /= np.sum(self.alpha_list)  # 进行规范化.
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1

        return y_pred.astype(_cml_precision.int)
