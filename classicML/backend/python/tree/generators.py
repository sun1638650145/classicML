"""classicML中树结构的生成器."""
import pandas as pd

from classicML.backend.training import get_criterion


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
            DecisionTreeClassifier实例.
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
            # 而是随机返回, 因此可能遇到决策树生成不一样的情况. 应该重写此函数.
            decision_tree.category = pd.value_counts(y).index[0]

            return decision_tree

        # 选择最优划分.
        feature_name, list_of_purity = self.criterion.optimal_division(x, y)

        decision_tree.feature_name = feature_name
        decision_tree.feature_index = list(self._x.columns).index(decision_tree.feature_name)
        current_feature_values = self._x.loc[:, decision_tree.feature_name]
        decision_tree.purity = list_of_purity[0]

        if len(list_of_purity) != 1:
            decision_tree.continuous = True
            decision_tree.dividing_point = list_of_purity[1]

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