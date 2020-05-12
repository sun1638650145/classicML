import numpy as np
import pandas as pd


def validation_accuaracy_after_split_pre(x_train, y_train, x_validation, y_validation, split_value=None):
    """使用预剪枝计算划分后的准确率"""

    if split_value is None:
        train_split = x_train
        validation_split = x_validation
    else:
        # 连续值需要按照划分点进行分组
        def split_group(x):
            if x >= split_value:
                return '>={:.3f}'.format(split_value)
            else:
                return '<{:.3f}'.format(split_value)

        train_split = x_train.map(split_group)
        validation_split = x_validation.map(split_group)

    majority_class_in_train = y_train.groupby(train_split).apply(lambda x: pd.value_counts(x).index[0])# 对应属性的取值的最多的映射表
    right_class_in_validation = y_validation.groupby(validation_split).apply(lambda x: np.sum(x == majority_class_in_train[x.name]))# 根据映射表统计验证集的样本数量
    return right_class_in_validation.sum() / y_validation.shape[0]


def validation_accuaracy_before_split_post(x_validation, y_validation, tree, split_value=None):
    """使用后剪枝划分前的准确率"""

    # work flow
    # 验证集符合和种类条件一样结点的集合
    # 找出符合划分属性的集合 lambda x: np.sum(x == tree.subtree[x.name].leaf_class
    # 找出和决策结点下的叶结点一致的样本 validation/validation_split
    # 分组 y_validation.groupby()
    if split_value is None:
        validation = x_validation.loc[:, tree.feature_name]
        right_class_in_validation = y_validation.groupby(validation).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))
    else:
        # 连续值需要按照划分点进行分组
        def split_group(x):
            if x >= tree.split_value:
                return '>={:.3f}'.format(tree.split_value)
            else:
                return '<{:.3f}'.format(tree.split_value)

        validation_split = x_validation.loc[:, tree.feature_name].map(split_group)
        right_class_in_validation = y_validation.groupby(validation_split).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))

    # 符合的样本的数量 / 总数
    return right_class_in_validation.sum() / y_validation.shape[0]


def reset_leaf(leaf_class, tree):
    """将分支结点设置为叶子结点"""

    tree.feature_name = None
    tree.feature_index = None

    tree.subtree = {} # 无分支

    tree.purity = None
    tree.is_continuous = None# 没有特征
    tree.split_value = None
    tree.is_leaf = True# 叶结点
    tree.leaf_class = leaf_class# 结点所属分类
    tree.leaf_num = 1
    tree.high = 0


def pre_pruning(x_train, y_train, x_validation, y_validation, tree=None):
    """
        进行预剪枝
        预剪枝是一种策略，生成后实现预剪枝是一种手段(自顶向下)
    """

    # 如果是决策树桩就没有必要剪枝
    if tree.is_leaf is True:
        return tree

    # 验证集为空
    if x_validation.empty is True:
        return tree

    # 计算当前的准确率
    most_class_in_train = pd.value_counts(y_train).index[0]# value_counts返回相等计数字符串的不一致顺序 pandas本身的bug
    before_split_accuracy = np.mean(y_validation == most_class_in_train)

    if tree.is_continuous is False:
        # 离散值
        after_split_accuaracy = validation_accuaracy_after_split_pre(x_train[tree.feature_name],
                                                                     y_train,
                                                                     x_validation[tree.feature_name],
                                                                     y_validation)

        if before_split_accuracy >= after_split_accuaracy:
            # 当前结点的划分准确率(泛化性能)没有提高 设置为叶结点
            reset_leaf(pd.value_counts(y_train).index[0], tree)
        else:
            max_high = -1
            tree.leaf_num = 0
            # 遍历属性集取值
            for key in tree.subtree.keys():
                this_part_train = x_train.loc[:, tree.feature_name] == key
                this_part_validation = x_validation.loc[:, tree.feature_name] == key
                tree.subtree[key] = pre_pruning(x_train[this_part_train],
                                                y_train[this_part_train],
                                                x_validation[this_part_validation],
                                                y_validation[this_part_validation],
                                                tree.subtree[key])

                if tree.subtree[key].high > max_high:
                    max_high = tree.subtree[key].high
                tree.leaf_num += tree.subtree[key].leaf_num
            tree.high = max_high + 1
    else:
        # 连续值
        after_split_accuaracy = validation_accuaracy_after_split_pre(x_train[tree.feature_name],
                                                                     y_train,
                                                                     x_validation[tree.feature_name],
                                                                     y_validation,
                                                                     tree.split_value)

        if before_split_accuracy >= after_split_accuaracy:
            # 当前结点的划分准确率(泛化性能)没有提高 设置为叶结点
            reset_leaf(pd.value_counts(y_train).index[0], tree)
        else:
            # 泛化性能提高时 进行二分划分
            up_part_train = x_train.loc[:, tree.feature_name] >= tree.split_value
            up_part_validation = x_validation.loc[:, tree.feature_name] >= tree.split_value
            down_part_train = x_train.loc[:, tree.feature_name] < tree.split_value
            down_part_validation = x_validation.loc[:, tree.feature_name] < tree.split_value

            up_subtree = pre_pruning(x_train[up_part_train],
                                     y_train[up_part_train],
                                     x_validation[up_part_validation],
                                     y_validation[up_part_validation],
                                     tree.subtree['>={:.3f}'.format(tree.split_value)])
            down_subtree = pre_pruning(x_train[down_part_train],
                                       y_train[down_part_train],
                                       x_validation[down_part_validation],
                                       y_validation[down_part_validation],
                                       tree.subtree['<{:.3f}'.format(tree.split_value)])

            tree.subtree['>={:.3f}'.format(tree.split_value)] = up_subtree
            tree.subtree['<{:.3f}'.format(tree.split_value)] = down_subtree
            tree.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)
            tree.high = max(up_subtree.high, down_subtree.high) + 1

    return tree


def post_pruning(x_train, y_train, x_validation, y_validation, tree=None):
    """
        进行后剪枝
        后剪枝(在决策结点上)
    """

    # 如果是决策树桩就没有必要剪枝
    if tree.is_leaf is True:
        return tree

    # 验证集为空
    if x_validation.empty is True:
        return tree

    # 计算剪枝之后的准确率(就是在当前种类多的 剪枝后会标记为多数类)
    most_class_in_train = pd.value_counts(y_train).index[0]# value_counts返回相等计数字符串的不一致顺序 pandas本身的bug
    after_split_accuracy = np.mean(y_validation == most_class_in_train)

    if tree.is_continuous is False:
        # 离散值
        max_high = -1
        tree.leaf_num = 0

        is_all_leaf = True

        for key in tree.subtree.keys():
            this_part_train = x_train.loc[:, tree.feature_name] == key
            this_part_validation = x_validation.loc[:, tree.feature_name] == key
            tree.subtree[key] = post_pruning(x_train[this_part_train],
                                             y_train[this_part_train],
                                             x_validation[this_part_validation],
                                             y_validation[this_part_validation],
                                             tree.subtree[key])

            # 后剪枝后下层结构改变 重新计算叶结点数和高度
            if tree.subtree[key].high > max_high:
                max_high = tree.subtree[key].high + 1
            tree.leaf_num += tree.subtree[key].leaf_num

            if tree.subtree[key].is_leaf is False:
                is_all_leaf = False
        tree.high = max_high

        if is_all_leaf is True:
            # 全是叶结点 说明是一个决策结点 进行处理
            before_split_accuaracy = validation_accuaracy_before_split_post(x_validation, y_validation, tree)
            if after_split_accuracy > before_split_accuaracy:
                # 当前结点的划分准确率(泛化性能)提高 设置为叶结点
                reset_leaf(pd.value_counts(y_train).index[0], tree)
    else:
        # 连续值
        # 进行二分划分
        up_part_train = x_train.loc[:, tree.feature_name] >= tree.split_value
        up_part_validation = x_validation.loc[:, tree.feature_name] >= tree.split_value
        down_part_train = x_train.loc[:, tree.feature_name] < tree.split_value
        down_part_validation = x_validation.loc[:, tree.feature_name] < tree.split_value

        up_subtree = post_pruning(x_train[up_part_train],
                                  y_train[up_part_train],
                                  x_validation[up_part_validation],
                                  y_validation[up_part_validation],
                                  tree.subtree['>={:.3f}'.format(tree.split_value)])
        down_subtree = post_pruning(x_train[down_part_train],
                                    y_train[down_part_train],
                                    x_validation[down_part_validation],
                                    y_validation[down_part_validation],
                                    tree.subtree['<{:.3f}'.format(tree.split_value)])

        # 返回值赋给子树
        tree.subtree['>={:.3f}'.format(tree.split_value)] = up_subtree
        tree.subtree['<{:.3f}'.format(tree.split_value)] = down_subtree

        tree.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)
        tree.high = max(up_subtree.high, down_subtree.high) + 1

        if up_subtree.is_leaf and down_subtree.is_leaf:
            # 全是叶结点 说明是一个决策结点 进行处理
            before_split_accuracy = validation_accuaracy_before_split_post(x_validation, y_validation, tree, tree.split_value)

            if after_split_accuracy > before_split_accuracy:
                # 当前结点的划分准确率(泛化性能)提高 设置为叶结点
                reset_leaf(pd.value_counts(y_train).index[0], tree)

    return tree
