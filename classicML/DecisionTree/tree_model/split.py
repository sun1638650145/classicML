from .backend import *


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