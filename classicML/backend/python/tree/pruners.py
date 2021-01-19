"""classicML中的树的剪枝器."""
import numpy as np
import pandas as pd


class Pruner(object):
    """剪枝器基类.

    Attributes:
        name: str, default=None,
            剪枝器的名称.
    """
    def __init__(self, name=None):
        super(Pruner, self).__init__()
        self.name = name

    def __call__(self, x, y, x_validation, y_validation, tree):
        """进行剪枝操作.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
            x_validation: pandas.DataFrame, 剪枝使用的验证特征数据.
            y_validation: pandas.DataFrame, 剪枝使用的验证标签.
            tree: classicML.backend.tree._TreeNode实例,
                决策树.
        """
        raise NotImplementedError

    def calculation_accuracy(self, *args, **kwargs):
        """计算使用预(后)剪枝操作的之后(前)的准确率."""
        raise NotImplementedError


class PostPruner(Pruner):
    """后剪枝器.
    """
    def __init__(self, name='post'):
        super(PostPruner, self).__init__(name=name)

    def __call__(self, x, y, x_validation, y_validation, tree):
        """进行剪枝操作.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
            x_validation: pandas.DataFrame, 剪枝使用的验证特征数据.
            y_validation: pandas.DataFrame, 剪枝使用的验证标签.
            tree: classicML.backend.tree._TreeNode实例,
                决策树.
        """
        if tree.leaf:
            return tree

        if x_validation.empty:
            return tree

        # TODO(Steve R. Sun, tag:code): 函数value_counts在两类元素值相同时, 不会按照某种顺序排序.
        # 而是随机返回, 因此可能遇到决策树生成不一样的情况. 应该重写此函数.
        current_category = pd.value_counts(y).index[0]
        accuracy_after_pruning = np.mean(y_validation == current_category)

        if tree.continuous:
            # 使用二分法对连续值进行处理.
            greater_part_of_train = (x.loc[:, tree.feature_name] >= tree.dividing_point)
            less_part_of_train = (x.loc[:, tree.feature_name] < tree.dividing_point)
            greater_part_of_validation = (x_validation.loc[:, tree.feature_name] >= tree.dividing_point)
            less_part_of_validation = (x_validation.loc[:, tree.feature_name] < tree.dividing_point)

            greater_part_tree = self.__call__(x[greater_part_of_train],
                                              y[greater_part_of_train],
                                              x_validation[greater_part_of_validation],
                                              y_validation[greater_part_of_validation],
                                              tree.subtree['>= {:.3f}'.format(tree.dividing_point)])
            less_part_tree = self.__call__(x[less_part_of_train],
                                           y[less_part_of_train],
                                           x_validation[less_part_of_validation],
                                           y_validation[less_part_of_validation],
                                           tree.subtree['< {:.3f}'.format(tree.dividing_point)])
            tree.subtree['>= {:.3f}'.format(tree.dividing_point)] = greater_part_tree
            tree.subtree['< {:.3f}'.format(tree.dividing_point)] = less_part_tree

            # 重新计算叶子数目和高度.
            tree.num_of_leaves = (greater_part_tree.num_of_leaves + less_part_tree.num_of_leaves)
            tree.height = max(greater_part_tree.height, less_part_tree.height) + 1
            # 此结点是一个分支结点且在最底层, 需要进行处理.
            if greater_part_tree.leaf and less_part_tree.leaf:
                accuracy_before_pruning = self.calculation_accuracy(x_validation, y_validation, tree)
                if accuracy_after_pruning > accuracy_before_pruning:
                    tree.reset(pd.value_counts(y).index[0])
        else:
            max_height = -1
            tree.num_of_leaves = 0
            all_leaves = True

            for key in tree.subtree.keys():
                part_of_train = (x.loc[:, tree.feature_name] == key)
                part_of_validation = (x_validation.loc[:, tree.feature_name] == key)

                tree.subtree[key] = self.__call__(x[part_of_train],
                                                  y[part_of_train],
                                                  x_validation[part_of_validation],
                                                  y_validation[part_of_validation],
                                                  tree.subtree[key])

                # 重新计算叶子数目和高度.
                tree.num_of_leaves += tree.subtree[key].num_of_leaves
                if tree.subtree[key].height > max_height:
                    max_height = tree.subtree[key].height + 1

                if tree.subtree[key].leaf is False:
                    all_leaves = False

            tree.height = max_height

            # 此结点是一个分支结点且在最底层, 需要进行处理.
            if all_leaves:
                accuracy_before_pruning = self.calculation_accuracy(x_validation, y_validation, tree)
                if accuracy_after_pruning > accuracy_before_pruning:
                    tree.reset(pd.value_counts(y).index[0])

        return tree

    def calculation_accuracy(self, x_validation, y_validation, tree):
        """计算剪枝前的准确率.
        这里没有采取原文的做法, 而是只计算这一个分支的数据,
        因为不修剪其他分支, 其他分支当前的值不改变也就不会影响准确率的总体变化,
        这样不仅代码好实现, 而且同时显著减少计算的开销.

        Arguments:
            x_validation: pandas.DataFrame, 剪枝使用的验证特征数据.
            y_validation: pandas.DataFrame, 剪枝使用的验证标签.
            tree: classicML.backend.tree._TreeNode实例,
                决策树.
        """
        if tree.dividing_point is None:
            part_of_validation = x_validation.loc[:, tree.feature_name]
        else:
            def _divide(feature):
                """对连续值的样本进行二分组."""
                if feature >= tree.dividing_point:
                    return '>= {:.3f}'.format(tree.dividing_point)
                else:
                    return '< {:.3f}'.format(tree.dividing_point)

            part_of_validation = x_validation.loc[:, tree.feature_name].map(_divide)

        correct_samples = y_validation.groupby(part_of_validation).apply(
            lambda x: np.sum(x == tree.subtree[x.name].category)
        )

        accuracy = correct_samples.sum() / y_validation.shape[0]

        return accuracy


class PrePruner(Pruner):
    """预剪枝器.

    Notes:
        - 这里只取用了预剪枝算法的思想, 实际实现还是在决策树生成以后进行的剪枝操作,
          因为如果按照原文实现势必影响一个正常的决策树生成.
    """
    def __init__(self, name='pre'):
        super(PrePruner, self).__init__(name=name)

    def __call__(self, x, y, x_validation, y_validation, tree):
        """进行剪枝操作.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
            x_validation: pandas.DataFrame, 剪枝使用的验证特征数据.
            y_validation: pandas.DataFrame, 剪枝使用的验证标签.
            tree: classicML.backend.tree._TreeNode实例,
                决策树.
        """
        if tree.leaf:
            return tree

        if x_validation.empty:
            return tree

        # TODO(Steve R. Sun, tag:code): 函数value_counts在两类元素值相同时, 不会按照某种顺序排序.
        # 而是随机返回, 因此可能遇到决策树生成不一样的情况. 应该重写此函数.
        current_category = pd.value_counts(y).index[0]
        accuracy_before_pruning = np.mean(y_validation == current_category)

        if tree.continuous:
            accuracy_after_pruning = self.calculation_accuracy(x[tree.feature_name],
                                                               y,
                                                               x_validation[tree.feature_name],
                                                               y_validation,
                                                               tree)
            # 泛化性能没有提高不进行剪枝, 故设置为叶结点.
            if accuracy_before_pruning >= accuracy_after_pruning:
                tree.reset(pd.value_counts(y).index[0])
            else:
                # 使用二分法对连续值进行处理.
                greater_part_of_train = (x.loc[:, tree.feature_name] >= tree.dividing_point)
                less_part_of_train = (x.loc[:, tree.feature_name] < tree.dividing_point)
                greater_part_of_validation = (x_validation.loc[:, tree.feature_name] >= tree.dividing_point)
                less_part_of_validation = (x_validation.loc[:, tree.feature_name] < tree.dividing_point)

                less_part_tree = self.__call__(x[less_part_of_train],
                                               y[less_part_of_train],
                                               x_validation[less_part_of_validation],
                                               y_validation[less_part_of_validation],
                                               tree.subtree['< {:.3f}'.format(tree.dividing_point)])
                greater_part_tree = self.__call__(x[greater_part_of_train],
                                                  y[greater_part_of_train],
                                                  x_validation[greater_part_of_validation],
                                                  y_validation[greater_part_of_validation],
                                                  tree.subtree['>= {:.3f}'.format(tree.dividing_point)])
                tree.subtree['< {:.3f}'.format(tree.dividing_point)] = less_part_tree
                tree.subtree['>= {:.3f}'.format(tree.dividing_point)] = greater_part_tree

                # 重新计算叶子数目和高度.
                tree.height = max(greater_part_tree.height, less_part_tree.height) + 1
                tree.num_of_leaves = (greater_part_tree.num_of_leaves + less_part_tree.num_of_leaves)
        else:
            accuracy_after_pruning = self.calculation_accuracy(x[tree.feature_name],
                                                               y,
                                                               x_validation[tree.feature_name],
                                                               y_validation,
                                                               tree)
            # 泛化性能没有提高不进行剪枝, 故设置为叶结点.
            if accuracy_before_pruning >= accuracy_after_pruning:
                tree.reset(pd.value_counts(y).index[0])
            else:
                max_height = -1
                tree.num_of_leaves = 0

                for key in tree.subtree.keys():
                    part_of_train = (x.loc[:, tree.feature_name] == key)
                    part_of_validation = (x_validation.loc[:, tree.feature_name] == key)

                    tree.subtree[key] = self.__call__(x[part_of_train],
                                                      y[part_of_train],
                                                      x_validation[part_of_validation],
                                                      y_validation[part_of_validation],
                                                      tree.subtree[key])

                    # 重新计算叶子高度和数目.
                    if tree.subtree[key].height > max_height:
                        max_height = tree.subtree[key].height + 1
                    tree.num_of_leaves += tree.subtree[key].num_of_leaves

                tree.height = max_height + 1

        return tree

    def calculation_accuracy(self, x, y, x_validation, y_validation, tree):
        """计算预剪枝划分后的准确率.

        Arguments:
            x: pandas.DataFrame, 特征数据.
            y: pandas.DataFrame, 标签.
            x_validation: pandas.DataFrame, 剪枝使用的验证特征数据.
            y_validation: pandas.DataFrame, 剪枝使用的验证标签.
            tree: classicML.backend.tree._TreeNode实例,
                决策树.
        """
        # 筛选出分支结点属性对应的属性值.
        if tree.dividing_point:
            def _divide(feature):
                """对连续值的样本进行二分组."""
                if feature >= tree.dividing_point:
                    return '>= {:.3f}'.format(tree.dividing_point)
                else:
                    return '< {:.3f}'.format(tree.dividing_point)

            part_of_train = x.loc[:, tree.feature_name].map(_divide)
            part_of_validation = x_validation.loc[:, tree.feature_name].map(_divide)
        else:
            part_of_train = x
            part_of_validation = x_validation

        # 获取分支结点每个属性标记最多的类别.
        category_in_train_set = y.groupby(part_of_train).apply(lambda x_: pd.value_counts(x_).index[0])

        correct_samples = y_validation.groupby(part_of_validation).apply(
            lambda x_: np.sum(x_ == category_in_train_set[x_.name])
        )

        accuracy = correct_samples.sum() / y_validation.shape[0]

        return accuracy