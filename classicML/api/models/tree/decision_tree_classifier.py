from pickle import dumps, loads

import numpy as np
import pandas as pd

from classicML import _cml_precision
from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import get_pruner
from classicML.backend import io
from classicML.backend import tree


class DecisionStumpClassifier(BaseModel):
    """决策树桩分类器.

    Attributes:
        stump: 生成的决策树桩.
        generator: 生成决策树桩的实现算法.
    """
    def __init__(self):
        """初始化决策树桩分类器.
        """
        super(DecisionStumpClassifier, self).__init__()

        self.stump = None
        self.generator = tree.generators.DecisionStumpGenerator()

    def fit(self, x, y, **kwargs):
        """训练决策树桩分类器.

        Args:
            x: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.

        Return:
            DecisionStumpClassifier实例.
        """
        x = np.asarray(x, dtype=_cml_precision.float)
        y = np.asarray(y, dtype=_cml_precision.int)

        # 生成决策树桩分类器.
        self.stump = self.generator(x, y)

        return self

    def predict(self, x, **kwargs):
        """使用决策树桩分类器进行预测.

        Args:
            x: numpy.ndarray or array-like, 特征数据.

        Return:
            DecisionStumpClassifier预测的结果.
        """
        y_pred = np.ones(shape=len(x), dtype=_cml_precision.int)

        if self.stump.division_mode == 'gte':
            y_pred[x[:, self.stump.feature_index] >= self.stump.dividing_point] = -1  # 划分方式是大于等于时, 设置大于等于划分点的值为反例.
        else:
            y_pred[x[:, self.stump.feature_index] < self.stump.dividing_point] = -1  # 划分方式是小于时, 设置小于划分点的值为反例.

        return y_pred


class DecisionTreeClassifier(BaseModel):
    """决策树分类器.

    Attributes:
        attribute_name: list of name, default=None,
            属性的名称.
        tree: classicML.backend.tree._TreeNode实例,
            生成的决策树.
        criterion: {'gain', 'gini', 'entropy'}, default='gain',
            决策树学习的划分方式.
        generator: classicML.backend.tree.DecisionTreeGenerator实例,
            生成决策树的实现算法.
        pruner: classicML.backend.tree.Pruner实例,
            决策树的剪枝器.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self, attribute_name=None):
        """初始化决策树.

        Arguments:
             attribute_name: list of name, default=None,
                属性的名称.
        """
        super(DecisionTreeClassifier, self).__init__()
        self.attribute_name = attribute_name

        self.tree = None
        self.criterion = None
        self.generator = None
        self.pruner = None

        self.is_trained = False
        self.is_loaded = False

    def compile(self, criterion='gain', pruning=None):
        """编译决策树, 配置训练时使用的超参数.

        Arguments:
            criterion: {'gain', 'gini', 'entropy'}, default='gain',
                决策树学习的划分方式.
            pruning: {None, 'pre', 'post'}, default=None,
                是否对决策树进行剪枝操作.

        Raises:
            AttributeError: 参数错误.
        """
        if criterion not in ('gain', 'gini', 'entropy'):
            CLASSICML_LOGGER.error('请检查你使用的划分方式')
            raise AttributeError('参数错误')
        if pruning not in (None, 'pre', 'post'):
            CLASSICML_LOGGER.error('请检查你输入的操作名称')
            raise AttributeError('参数错误')

        self.criterion = criterion
        self.generator = tree.generators.DecisionTreeGenerator(criterion=self.criterion)
        self.pruner = get_pruner(pruning=pruning)

    def fit(self, x, y, x_validation=None, y_validation=None):
        """训练决策树分类器.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like,
                特征数据.
            y: numpy.ndarray or pandas.DataFrame, array-like,
                标签.
            x_validation: numpy.ndarray or pandas.DataFrame, array-like,
                剪枝使用的验证特征数据.
            y_validation: numpy.ndarray or pandas.DataFrame, array-like,
                剪枝使用的验证标签.

        Returns:
            DecisionTreeClassifier实例.

        Raises:
            AttributeError: 没有验证集.
        """
        if isinstance(x, np.ndarray):
            x = x.tolist()
            if self.attribute_name is None:
                CLASSICML_LOGGER.warn("属性名称缺失, 请使用pandas.DataFrame; 或检查 self.attributes_name")
        if (self.pruner is not None) and (x_validation is None or y_validation is None):
            CLASSICML_LOGGER.error("没有验证集, 无法对决策树进行剪枝")
            raise AttributeError('没有验证集')

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)
        self.generator._x = x

        y = pd.Series(y, dtype=_cml_precision.int)
        y.reset_index(drop=True, inplace=True)

        # 为验证数据添加属性信息.
        if x_validation is not None:
            x_validation = pd.DataFrame(x_validation, columns=self.attribute_name)
            x_validation.reset_index(drop=True, inplace=True)

            y_validation = pd.Series(y_validation, dtype=_cml_precision.int)
            y_validation.reset_index(drop=True, inplace=True)

        # 没有使用权重文件, 则生成决策树分类器.
        if self.is_loaded is False:
            self.tree = self.generator(x, y)

        # 进行剪枝.
        if self.pruner:
            self.tree = self.pruner(x, y, x_validation, y_validation, self.tree)

        # 标记训练完成
        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """使用决策树分类器进行预测.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like,
                特征数据.

        Returns:
            DecisionTreeClassifier预测的结果.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        # 修正数据类型.
        if isinstance(x, list):
            x = np.expand_dims(x, axis=0)
        elif isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, pd.Series):
            x = np.expand_dims(x.values, axis=0)
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        y_pred = np.zeros(shape=len(x), dtype=_cml_precision.int)
        for i, feature in enumerate(x):
            y_pred[i] = self._predict(feature, self.tree)

        return y_pred

    def _predict(self, x, decision_tree):
        """通过递归决策树预测结果.

        Arguments:
            x: numpy.ndarray, 特征数据.
            decision_tree: classicML.backend.tree._TreeNode, 决策树实例.
        """
        if decision_tree.leaf:
            return decision_tree.category

        if decision_tree.continuous:
            if x[decision_tree.feature_index] >= decision_tree.dividing_point:
                return self._predict(x, decision_tree.subtree['>= {:.3f}'.format(decision_tree.dividing_point)])
            else:
                return self._predict(x, decision_tree.subtree['< {:.3f}'.format(decision_tree.dividing_point)])
        else:
            return self._predict(x, decision_tree.subtree[x[decision_tree.feature_index]])

    def score(self, x, y):
        """在预测模式下计算准确率.

        Arguments:
            x: array-like, 特征数据.
            y: array-like, 标签.

        Returns:
            当前的准确率.
        """
        return super(DecisionTreeClassifier, self).score(x, y)

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
                                                   model_name='DecisionTreeClassifier')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.criterion = compile_ds.attrs['criterion']
            if self.pruner is not None:
                self.pruner = get_pruner(compile_ds.attrs['pruning'])

            self.tree = loads(weights_ds.attrs['tree'].tobytes())

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

        References:
            - [如何存储原始的二进制数据](https://docs.h5py.org/en/2.3/strings.html)

        Notes:
            模型将不会保存关于优化器的超参数.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='w',
                                                   model_name='DecisionTreeClassifier')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['criterion'] = self.criterion
            if self.pruner is not None:
                compile_ds.attrs['pruning'] = self.pruner.name
            weights_ds.attrs['tree'] = np.void(dumps(self.tree))
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')


class TwoLevelDecisionTreeClassifier(BaseModel):
    """2层决策树分类器.

    Attributes:
        tree: 生成的2层决策树.
        generator: 生成2层决策树的实现算法.
    """
    def __init__(self):
        """初始化2层决策树分类器.
        """
        super(TwoLevelDecisionTreeClassifier, self).__init__()

        self.tree = None
        self.generator = tree.generators.TwoLevelDecisionTreeGenerator()

    def fit(self, x, y, **kwargs):
        """训练2层决策树分类器.

        Args:
            x: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            kwargs:
                sample_distribution: numpy.ndarray,
                    样本分布.

        Return:
            TwoLevelDecisionTreeClassifier实例.
        """
        sample_distribution = kwargs['sample_distribution']

        x = np.asarray(x, dtype=_cml_precision.float)
        y = np.asarray(y, dtype=_cml_precision.int)

        # 生成2层决策树分类器.
        self.tree = self.generator(x, y, sample_distribution)

        return self

    def predict(self, x, **kwargs):
        """使用2层决策树分类器进行预测.

        Args:
            x: numpy.ndarray or array-like, 特征数据.

        Return:
            TwoLevelDecisionTreeClassifier预测的结果.
        """
        y_pred = np.zeros(shape=len(x), dtype=_cml_precision.int)
        for i, feature in enumerate(x):
            y_pred[i] = self._predict(feature, self.tree)

        return y_pred

    def _predict(self, x, decision_tree):
        """通过递归决策树预测结果.

        Args:
            x: numpy.ndarray, 特征数据.
            decision_tree: classicML.backend.tree._TreeNode, 决策树实例.
        """
        if decision_tree.leaf:
            return decision_tree.category

        if x[decision_tree.feature_index] >= decision_tree.dividing_point:
            return self._predict(x, decision_tree.subtree['upper_tree'])
        else:
            return self._predict(x, decision_tree.subtree['lower_tree'])
