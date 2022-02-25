"""classicML中树结构的生成器."""
import os

import numpy as np
import pandas as pd

from classicML import _cml_precision
from classicML.backend.training import get_criterion

if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.metrics import BinaryAccuracy
else:
    from classicML.backend.python.metrics import BinaryAccuracy


class _DecisionStump(object):
    """决策树桩.

    Attributes:
        feature_index: int, default=-1,
            划分的类别的下标.
        dividing_point: float, default=None,
            划分点的值.
        division_mode: {'gte', 'le'}, default='',
            划分模式.
    """
    def __init__(self):
        self.feature_index = -1
        self.dividing_point = None
        self.division_mode = ''


class _TreeNode(object):
    """树的结点.

    Attributes:
        num_of_leaves: int, default=None,
            叶结点的数量, None表示没有叶结点, 即本身就是叶结点.
        leaf: bool, default=False,
            是否为一个叶结点.
        height: int, default=-1,
            当前结点的高度.
        category: str, default='',
            当前结点的表示的类别.
        feature_name: str, default='',
            当前结点表示划分的类别.
        feature_index: int, default=None,
            当前结点表示划分的类别的下标.
        purity: float, default=None,
            当前结点的纯度.
        continuous: bool, default=False,
            是否是连续属性.
        subtree: dict, default={},
            属性的分支, 即子树.
        dividing_point: float, default=None,
            当前的结点是连续值时, 划分点的值.
    """
    def __init__(self):
        self.num_of_leaves = None
        self.leaf = False
        self.height = -1
        self.category = ''
        self.feature_name = ''
        self.feature_index = None
        self.purity = None
        self.continuous = False
        self.subtree = dict()
        self.dividing_point = None

    def reset(self, category):
        """重置结点.

        Arguments:
            category: str, 当前结点的表示的类别.
        """
        self.num_of_leaves = 1
        self.height = 0
        self.category = category
        self.leaf = True
        self.feature_name = ''
        self.feature_index = None
        self.purity = None
        self.continuous = False
        self.subtree = dict()
        self.dividing_point = None


class TreeGenerator(object):
    """树生成器的基类.

    Attributes:
        name: str, 生成器的名称.
        criterion: {'gain', 'gini', 'entropy'}, default='gain',
            决策树学习的划分方式.
    """
    def __init__(self, name=None, criterion=None):
        """初始化生成器.

        Arguments:
            name: str, 生成器的名称.
            criterion: {'gain', 'gini', 'entropy'}, default='gain',
                决策树学习的划分方式.
        """
        super(TreeGenerator, self).__init__()
        self.name = name
        self.criterion = get_criterion(criterion)

    def __call__(self, *args, **kwargs):
        """功能实现."""
        return self.tree_generate(*args, **kwargs)

    def tree_generate(self, *args, **kwargs):
        """树的生成实现."""
        raise NotImplementedError


class TwoLevelDecisionTreeGenerator(TreeGenerator):
    """2层决策树生成器.
    """
    def __init__(self, name='2-level_decision_tree_generator', criterion='weighted_gini'):
        super(TwoLevelDecisionTreeGenerator, self).__init__(name, criterion)

    def tree_generate(self, D, y, sample_distribution, height=0):
        """生成决策树.

        Args:
            D: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            sample_distribution: numpy.ndarray,
                样本分布.
            height: int, default=0,
                决策树的高度.

        Return:
            _TreeNode树结点实例.
        """
        decision_tree = _TreeNode()
        decision_tree.height = height

        # 高度为2或者样本少于2.
        if height == 2 or len(D) <= 2:
            positive_weight = np.sum(sample_distribution[y == 1])
            negative_weight = np.sum(sample_distribution[y == -1])

            # 设置为叶子结点.
            decision_tree.leaf = True
            if positive_weight > negative_weight:
                decision_tree.category = 1
            else:
                decision_tree.category = -1

            return decision_tree

        decision_tree.feature_index, decision_tree.dividing_point = (
            self.choose_feature_to_divide(D, y, sample_distribution)
        )

        D_upper = D[:, decision_tree.feature_index] >= decision_tree.dividing_point
        D_lower = D[:, decision_tree.feature_index] < decision_tree.dividing_point

        # 递归产生子树.
        decision_tree.subtree['upper_tree'] = self.tree_generate(
            D[D_upper, :], y[D_upper], sample_distribution[D_upper], height + 1)
        decision_tree.subtree['lower_tree'] = self.tree_generate(
            D[D_lower, :], y[D_lower], sample_distribution[D_lower], height + 1)

        return decision_tree

    def choose_feature_to_divide(self, D, y, sample_distribution):
        """选择最优划分.

        Args:
            D: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            sample_distribution: numpy.ndarray,
                样本分布.

        Returns:
            当前结点划分属性的索引和划分点的值.
        """
        criterion_value = float('inf')
        dividing_point = None
        feature_index = -1

        for i in range(D.shape[1]):
            current_criterion_value, current_dividing_point = self.criterion.get_value(D[:, i], y, sample_distribution)
            if criterion_value > current_criterion_value:
                criterion_value = current_criterion_value  # 更新迭代不可删.
                dividing_point = current_dividing_point
                feature_index = i

        return feature_index, dividing_point


class DecisionStumpGenerator(TreeGenerator):
    """决策树桩生成器.

    Attributes:
        name: str, 生成器的名称.
    """
    def __init__(self, name='decision_stump_generator'):
        """初始化生成器.

        Args:
            name: str, 生成器的名称.
        """
        self.name = name

    def tree_generate(self, D, y):
        """生成决策树桩.

        Args:
            D: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.

        Return:
            _DecisionStump实例.
        """
        stump = _DecisionStump()
        error = _cml_precision.float(np.inf)

        # TODO(Steve Sun, tag:code): 时间复杂度O(N^3)过高.
        for i in range(D.shape[1]):
            # 构建连续值候选划分点集.
            D_ = np.sort(D[:, i])
            T_set = [
                ((D_[i] + D_[i + 1]) / 2) for i in range(len(D_) - 1)
            ]

            for division_mode in ('gte', 'le'):
                for t in T_set:
                    current_error = self.evaluate(D, y, i, t, division_mode)

                    if current_error < error:
                        stump.feature_index = i
                        stump.dividing_point = t
                        stump.division_mode = division_mode
                        error = current_error

        return stump

    @staticmethod
    def evaluate(D, y, column, dividing_point, division_mode):
        """对当前的划分方式进行评估.

        Args:
            D: numpy.ndarray or array-like, 特征数据.
            y: numpy.ndarray or array-like, 标签.
            column: int, 划分的类别的下标.
            dividing_point: float, 划分点的值.
            division_mode: {'gte', 'le'}, 划分模式.

        Return:
            当前的划分方式评估的结果.
        """
        y_pred = np.ones(shape=(len(D)), dtype=_cml_precision.int)
        if division_mode == 'gte':
            y_pred[D[:, column] >= dividing_point] = -1  # 划分方式是大于等于时, 设置大于等于划分点的值为反例.
        else:
            y_pred[D[:, column] < dividing_point] = -1  # 划分方式是小于时, 设置小于划分点的值为反例.

        y_pred = y_pred.reshape(-1, 1)
        error = 1 - BinaryAccuracy()(y_pred, y)

        return error


class DecisionTreeGenerator(TreeGenerator):
    """决策树生成器.

    Attributes:
        _x: pandas.DataFrame,
            未经训练的原始特征数据.
            保存一份原始的数据必然会占用两倍的内存, 但是划分最优属性后, 应该为所有出现过的属性值进行生成分支(子结点).
    """
    def __init__(self, name='decision_tree_generator', criterion=None):
        super(DecisionTreeGenerator, self).__init__(name=name,
                                                    criterion=criterion)

        self._x = None

    def tree_generate(self, x, y):
        """生成决策树.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.

        Returns:
            _TreeNode树结点实例.
        """
        decision_tree = _TreeNode()
        decision_tree.num_of_leaves = 0

        # 样本属于同一个类别.
        if y.nunique() == 1:
            decision_tree.num_of_leaves += 1
            decision_tree.leaf = True
            decision_tree.height = 0
            decision_tree.category = y.values[0]

            return decision_tree

        # 属性被测试完.
        if x.empty:
            decision_tree.num_of_leaves += 1
            decision_tree.leaf = True
            decision_tree.height = 0
            # TODO(Steve R. Sun, tag:code): 函数value_counts在两类元素值相同时, 不会按照某种顺序排序.
            #  而是随机返回, 因此可能遇到决策树生成不一样的情况. 应该重写此函数.
            decision_tree.category = pd.value_counts(y).index[0].astype(_cml_precision.int)

            return decision_tree

        # 选择最优划分.
        feature_name, list_of_purity = self.criterion.optimal_division(x, y)

        decision_tree.feature_name = feature_name
        decision_tree.feature_index = list(self._x.columns).index(decision_tree.feature_name)
        current_feature_values = self._x.loc[:, decision_tree.feature_name]
        decision_tree.purity = _cml_precision.float(list_of_purity[0])

        if len(list_of_purity) != 1:
            decision_tree.continuous = True
            decision_tree.dividing_point = _cml_precision.float(list_of_purity[1])

            # 使用二分法对连续值进行处理.
            greater_part = '>= {:.3f}'.format(decision_tree.dividing_point)
            less_part = '< {:.3f}'.format(decision_tree.dividing_point)
            decision_tree.subtree[greater_part] = self.tree_generate(
                x.loc[current_feature_values >= decision_tree.dividing_point],
                y.loc[current_feature_values >= decision_tree.dividing_point])
            decision_tree.subtree[less_part] = self.tree_generate(
                x.loc[current_feature_values < decision_tree.dividing_point],
                y.loc[current_feature_values < decision_tree.dividing_point])

            decision_tree.num_of_leaves += (decision_tree.subtree[greater_part].num_of_leaves
                                            + decision_tree.subtree[less_part].num_of_leaves)
            decision_tree.height = (decision_tree.subtree[greater_part].height
                                    + decision_tree.subtree[less_part].height) + 1
        else:
            decision_tree.continuous = False

            feature_values = pd.unique(current_feature_values)
            sub_x = x.drop(decision_tree.feature_name, axis=1)  # 最优属性相当于已被使用掉.

            max_height = -1
            # 为每个属性值生成一个分支.
            for feature_value in feature_values:
                if y[current_feature_values == feature_value].empty is True:
                    decision_tree.subtree[feature_value] = self.tree_generate(
                        sub_x.loc[current_feature_values == feature_value],
                        y)  # 如果为空, 理论上标记是数据集中出现的最多的样本, 但是这样并不好通过递归实现, 于是此处传入父节点的样本标记.
                else:
                    decision_tree.subtree[feature_value] = self.tree_generate(
                        sub_x.loc[current_feature_values == feature_value],
                        y.loc[current_feature_values == feature_value])

                # 更新子树的高度.
                if decision_tree.subtree[feature_value].height > max_height:
                    max_height = decision_tree.subtree[feature_value].height

                decision_tree.num_of_leaves += decision_tree.subtree[feature_value].num_of_leaves

            decision_tree.height = max_height + 1

        return decision_tree
