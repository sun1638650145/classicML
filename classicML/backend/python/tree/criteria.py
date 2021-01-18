"""classicML中决策树的划分标准."""
import os
import numpy as np
import pandas as pd

if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.ops import cc_type_of_target as type_of_target
else:
    from classicML.backend.python.ops import type_of_target


class Criterion(object):
    """划分标准基类."""
    def __init__(self, name=None):
        """
        Arguments:
            name: str, default=None,
                划分标准的名称.
        """
        super(Criterion, self).__init__()
        self.name = name

    def __call__(self, D):
        """划分标准算法实现.

        Arguments:
            D: pandas.Series,
                需要计算的数据集.
        """
        raise NotImplementedError

    def get_value(self, *args, **kwargs):
        """计算划分标准的值.

        Arguments:
            args:
                D: pandas.Series, 需要计算的数据集.
                y: pandas.DataFrame, 对应的标签.
                continuous: bool, 是否是连续属性.
        """
        D, y, continuous = args[0], args[1], args[2]

        num_of_labels = y.shape[0]
        unique_features = pd.unique(D)

        if continuous:
            unique_features.sort()
            criterion_value = float('inf')
            dividing_point = None

            # 构建连续值候选划分点集.
            T_set = [
                ((unique_features[i] + unique_features[i + 1]) / 2) for i in range(len(unique_features) - 1)
            ]

            for t in T_set:
                Dv_up = y[D >= t]
                Dv_down = y[D < t]
                current_criterion_value = (Dv_up.shape[0] / num_of_labels * self.__call__(Dv_up)
                                           + Dv_down.shape[0] / num_of_labels * self.__call__(Dv_down))

                if criterion_value > current_criterion_value:
                    criterion_value = current_criterion_value
                    dividing_point = t

            return [criterion_value, dividing_point]
        else:
            criterion_value = 0
            for feature in unique_features:
                Dv = y[feature == D]
                criterion_value += Dv.shape[0] / num_of_labels * self.__call__(Dv)

            return [criterion_value]

    def optimal_division(self, x, y):
        """最优的划分属性.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
        """
        attribute_name = x.columns

        # 初始化为list形式是为了保证比较的时候可以都进行数值的比较,
        # 如果遇到连续值更新的时候会返回值和划分点.
        best_criterion_value = [float('inf')]
        best_attribute = None

        for attribute in attribute_name:
            # 需要先判断是连续值还是离散值.
            continuous = (type_of_target(np.asarray(x[attribute])) == 'continuous')
            # 计算基尼指数.
            criterion_value = self.get_value(x[attribute], y, continuous)
            if best_criterion_value[0] > criterion_value[0]:
                best_criterion_value = criterion_value
                best_attribute = attribute

        return best_attribute, best_criterion_value


class Entropy(Criterion):
    """信息熵."""
    def __init__(self, name='entropy'):
        super(Entropy, self).__init__(name=name)

    def __call__(self, D):
        """计算信息熵.

        Arguments:
            D: pandas.Series,
                需要计算的数据集.
        """
        pk = pd.value_counts(D) / D.shape[0]
        value = np.sum(-pk * np.log2(pk))

        return value


class Gain(Entropy):
    """信息增益."""
    def __init__(self, name='gain'):
        super(Gain, self).__init__(name=name)

    def get_value(self, D, y, D_entropy, continuous):
        """计算信息增益.

        Arguments:
            D: pandas.Series, 需要计算的数据集.
            y: pandas.DataFrame, 对应的标签.
            D_entropy: float, 整个数据集的信息熵.
            continuous: bool, 是否是连续属性.
        """
        if continuous:
            entropy, dividing_point = super(Gain, self).get_value(D, y, continuous)
            # 计算信息增益.
            gain = D_entropy - entropy

            return [gain, dividing_point]
        else:
            entropy = super(Gain, self).get_value(D, y, continuous)
            gain = D_entropy - entropy

            return [gain]

    def optimal_division(self, x, y):
        """最优的划分属性.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
        """
        attribute_name = x.columns

        # 初始化为list形式是为了保证比较的时候可以都进行数值的比较,
        # 如果遇到连续值更新的时候会返回值和划分点.
        best_gain_value = [float('-inf')]
        best_attribute = None
        # 计算整个数据集的信息熵.
        D_entropy = self.__call__(y)

        for attribute in attribute_name:
            # 需要先判断是连续值还是离散值.
            continuous = (type_of_target(np.asarray(x[attribute])) == 'continuous')
            # 计算信息增益.
            gain_value = self.get_value(x[attribute], y, D_entropy, continuous)
            if best_gain_value[0] < gain_value[0]:
                best_gain_value = gain_value
                best_attribute = attribute

        return best_attribute, best_gain_value


class Gini(Criterion):
    """基尼指数."""
    def __init__(self, name='gini'):
        super(Gini, self).__init__(name=name)

    def __call__(self, D):
        """计算基尼指数.

        Arguments:
            D: pandas.Series,
                需要计算的数据集.
        """
        pk = pd.value_counts(D) / D.shape[0]
        value = 1 - np.sum(pk ** 2)

        return value