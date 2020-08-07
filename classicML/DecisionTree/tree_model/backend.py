import pandas as pd
import numpy as np


def information_entropy(D):
    """计算信息熵"""
    pk = pd.value_counts(D) / D.shape[0] # 计算各类的占比
    ent = np.sum(-pk * np.log2(pk))

    return ent


def gini(D):
    """计算基尼指数"""
    pk = pd.value_counts(D) / D.shape[0] # 计算各类的占比
    gini = 1 - np.sum(pk * pk)

    return gini


def gain(feature, y, ent, is_continuous=False):
    """计算信息增益"""

    num = y.shape[0]# 样本总数
    unique_features = pd.unique(feature)# 去重以后得到属性列表

    if is_continuous is False:
        # 离散值

        ent_feature = 0
        for unique_feature in unique_features:
            D_v = y[feature == unique_feature]
            ent_feature += D_v.shape[0] / num * information_entropy(D_v)

        gain = ent - ent_feature

        return [gain]
    else:
        # 连续值

        unique_features.sort()
        # 建立候选划分点t的集合
        t_set = [((unique_features[i] + unique_features[i + 1]) / 2) for i in range(len(unique_features) - 1)]
        # 选择最优分割点
        best_split_point = None
        best_ent = float('inf')
        for t in t_set:
            D_v_up = y[feature >= t]
            D_v_down = y[feature < t]
            ent_feature = D_v_up.shape[0] / num * information_entropy(D_v_up) + D_v_down.shape[0] / num * information_entropy(D_v_down)

            # 选择信息增益最大的点 就是ent最小
            if ent_feature < best_ent:
                best_ent = ent_feature
                best_split_point = t

        gain = ent - best_ent

        return [gain, best_split_point]


def gini_index(feature, y, is_continuous=False):
    """计算属性的基尼指数"""

    num = y.shape[0]# 样本总数
    unique_features = pd.unique(feature)# 去重以后得到属性列表

    if is_continuous is False:
        # 离散值

        gini_index = 0
        for unique_feature in unique_features:
            D_v = y[feature == unique_feature]
            gini_index += D_v.shape[0] / num * gini(D_v)

        return [gini_index]
    else:
        # 连续值

        unique_features.sort()
        # 建立候选划分点t的集合
        t_set = [((unique_features[i] + unique_features[i + 1]) / 2) for i in range(len(unique_features) - 1)]
        # 选择最优分割点
        best_split_point = None
        gini_index = float('inf')
        for t in t_set:
            D_v_up = y[feature >= t]
            D_v_down = y[feature < t]
            gini_index_feature = D_v_up.shape[0] / num * gini(D_v_up) + D_v_down.shape[0] / num * gini(D_v_down)

            # 选择基尼指数
            if gini_index_feature < gini_index:
                gini_index = gini_index_feature
                best_split_point = t

        return [gini_index, best_split_point]


def entropy(feature, y, is_continuous=False):
    """计算信息熵"""

    num = y.shape[0]# 样本总数
    unique_features = pd.unique(feature)# 去重以后得到属性列表

    if is_continuous is False:
        # 离散值

        ent_sum = 0
        for unique_feature in unique_features:
            D_v = y[feature == unique_feature]
            ent_sum += D_v.shape[0] / num * information_entropy(D_v)

        return [ent_sum]
    else:
        # 连续值

        unique_features.sort()
        # 建立候选划分点t的集合
        t_set = [((unique_features[i] + unique_features[i + 1]) / 2) for i in range(len(unique_features) - 1)]
        # 选择最优分割点
        best_split_point = None
        best_ent = float('inf')
        for t in t_set:
            D_v_up = y[feature >= t]
            D_v_down = y[feature < t]
            ent_sum = (D_v_up.shape[0] / num * information_entropy(D_v_up) + D_v_down.shape[0] / num * information_entropy(D_v_down))

            # 选择信息熵最小
            if best_ent > ent_sum:
                best_ent = ent_sum
                best_split_point = t

        return [best_ent, best_split_point]


def type_of_target(y):
    """
    判断输入的类型

    Parameters
    ----------
    y: array-like，y的维度必须是1-D或者2-D

    Returns
    -------
    输入数据的类型: str
        'continuous': 所有元素都是浮点数，且不是对应整数的浮点数
        'binary': 所有元素只有两个离散值，类型不限制
        'multiclass': 所有元素不只有两个离散值，类型不限制
        'multilabel': 元素标签不唯一，类型不限制
        'unknown': 其他情况一律返回未知
    """

    y = np.asarray(y)
    if y.dtype == object or np.ndim(y) > 2:
        return 'unknown'

    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        return 'continuous'

    if y.shape[1] == 1:
        if len(np.unique(y)) == 2:
            return 'binary'
        elif len(np.unique(y)) > 2:
            return 'multiclass'

    if np.ndim(y) == 2 and len(np.unique(y)) >= 2:
        return 'multilabel'

    return 'unknown'