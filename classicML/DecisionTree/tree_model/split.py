import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


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


def choose_best_split_with_gain(x, y):
    """划分选择方式是信息增益"""

    features = x.columns

    # 初始化最优划分属性和信息增益的值
    best_feature_name = None
    best_gain = [float('-inf')]

    ent_D = information_entropy(y)
    for feature in features:
        # 判断是否是连续值
        is_continuous = type_of_target(x[feature]) == 'continuous'
        # 计算信息增益
        gain_value = gain(x[feature], y, ent_D, is_continuous)
        if best_gain[0] < gain_value[0]:
            best_gain = gain_value
            best_feature_name = feature

    return best_feature_name, best_gain


def choose_best_split_with_gini(x, y):
    """划分选择方式是基尼指数"""

    features = x.columns

    # 初始化最优划分属性和基尼指数的值
    best_feature_name = None
    best_gini_index = [float('inf')]

    for feature in features:
        # 判断是否是连续值
        is_continuous = type_of_target(x[feature]) == 'continuous'
        # 计算基尼指数
        gini_index_value = gini_index(x[feature], y, is_continuous)
        if best_gini_index[0] > gini_index_value[0]:
            best_gini_index = gini_index_value
            best_feature_name = feature

    return best_feature_name, best_gini_index


def choose_best_split_with_entropy(x, y):
    """划分选择方式是信息熵"""

    features = x.columns

    # 初始化最优划分属性和信息熵的值
    best_feature_name = None
    best_entropy = [float('inf')]

    for feature in features:
        # 判断是否是连续值
        is_continuous = type_of_target(x[feature]) == 'continuous'
        # 计算信息熵
        entropy_value = entropy(x[feature], y, is_continuous)
        if best_entropy[0] > entropy_value[0]:
            best_entropy = entropy_value
            best_feature_name = feature

    return best_feature_name, best_entropy