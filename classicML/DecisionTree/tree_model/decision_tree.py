from .split import *
from .pruning import *


class Node:
    def __init__(self):
        self.feature_name = None
        self.feature_index = None

        self.subtree = {}  # 属性值分支

        self.purity = None
        self.is_continuous = False  # 是否是离散的特征
        self.split_value = None
        self.is_leaf = False  # 是否是叶结点
        self.leaf_class = None  # 结点所属分类
        self.leaf_num = None  # 表示无 树桩1
        self.high = -1  # 表示无 树桩0


class DecisionTree:
    """生成一个决策树"""

    def __init__(self):
        pass

    def compile(self, critertion='gain', pruning=None, feature_attr=None):
        assert critertion in ('gain', 'gini', 'entropy')
        assert pruning in (None, 'pre', 'post')

        self.critertion = critertion
        self.pruning = pruning
        self.feature_attr = feature_attr
        try:
            if self.feature_attr is None:
                raise UserWarning
        except UserWarning:
            print("Warning: 如果feature_attr为None, 请保证输出的x为pandas.DataFrame")

    def fit(self, x, y, x_validation=None, y_validation=None):

        # 检查剪枝条件是否满足
        if self.pruning is not None:
            if (x_validation is None) or (y_validation is None):
                raise Exception("没有验证集不能进行剪枝")

        # 添加index列
        x = pd.DataFrame(x, columns=self.feature_attr)
        y = pd.Series(y)
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # 属性的特征数必须是从全训练集统计
        self.original_x = x

        if (x_validation is not None) and (y_validation is not None):
            x_validation = pd.DataFrame(x_validation, columns=self.feature_attr)
            y_validation = pd.Series(y_validation)
            x_validation.reset_index(drop=True, inplace=True)
            y_validation.reset_index(drop=True, inplace=True)

        # 生成决策树
        self.columns = list(x.columns)
        self.tree = self.tree_generate(x, y)

        # 进行剪枝
        if self.pruning == 'pre':
            pre_pruning(x, y, x_validation, y_validation, self.tree)
        elif self.pruning == 'post':
            post_pruning(x, y, x_validation, y_validation, self.tree)

        return self

    def predict(self, x):
        """进行预测 x可以是一个list、Series或者DataFrames"""

        if not hasattr(self, 'tree'):
            raise Exception("你必须先进行训练")

        if type(x) is list:
            return list(self.predict_engine(x))
        elif x.ndim == 1:
            return list(self.predict_engine(x))
        else:
            return pd.Series.tolist(x.apply(self.predict_engine, axis=1))

    def predict_engine(self, x, subtree=None):
        if subtree is None:
            subtree = self.tree

        if subtree.is_leaf:
            return subtree.leaf_class

        if subtree.is_continuous is False:
            return self.predict_engine(x, subtree.subtree[x[subtree.feature_index]])
        else:
            if x[subtree.feature_index] >= subtree.split_value:
                return self.predict_engine(x, subtree.subtree['>={:.3f}'.format(subtree.split_value)])
            else:
                return self.predict_engine(x, subtree.subtree['<{:.3f}'.format(subtree.split_value)])

    def tree_generate(self, x, y):
        tree = Node()# 1
        # 首先划分为决策树桩
        tree.leaf_num = 0

        if y.nunique() == 1:# 2
            tree.leaf_class = y.values[0]# 3 标记种类
            tree.is_leaf = True# 3 标记叶结点
            tree.high = 0
            tree.leaf_num += 1

            return tree

        if x.empty is True:# 5/11
            tree.leaf_class = pd.value_counts(y).index[0]# 6 标记为最多的种类
            tree.is_leaf = True# 6/12 标记叶结点
            tree.high = 0
            tree.leaf_num += 1

            return tree

        best_feature_name, purity_list = self.choose_best_split(x, y)# 8 选择最优划分
        tree.feature_name = best_feature_name
        tree.purity = purity_list[0]
        tree.feature_index = self.columns.index(best_feature_name)
        feature_values = self.original_x.loc[:, best_feature_name]# pandas切片出最佳特征对应的列

        if len(purity_list) == 1:
            # 离散值
            tree.is_continuous = False

            attr_list = pd.unique(feature_values)# 去除重复，得到属性列表
            sub_x = x.drop(best_feature_name, axis=1)# 对于离散值的计算，再次划分去除以选择的最优特征对应的列

            max_high = -1
            for attr in attr_list:# 9
                if y[feature_values == attr].empty is not True:
                    tree.subtree[attr] = self.tree_generate(sub_x[feature_values == attr], y[feature_values == attr])# 10/14
                else:
                    # 如果是空，应该传入父结点
                    tree.subtree[attr] = self.tree_generate(sub_x[feature_values == attr], y)
                if tree.subtree[attr].high > max_high:
                    max_high = tree.subtree[attr].high
                tree.leaf_num += tree.subtree[attr].leaf_num

            tree.high = max_high + 1
        elif len(purity_list) == 2:
            # 连续值 二分后分别计算purity
            tree.is_continuous = True
            tree.split_value = purity_list[1]# 选Ent小

            # 二分法离散化
            up_part = '>={:.3f}'.format(tree.split_value)
            down_part = '<{:.3f}'.format(tree.split_value)
            tree.subtree[up_part] = self.tree_generate(x[feature_values >= tree.split_value], y[feature_values >= tree.split_value])# 10/14
            tree.subtree[down_part] = self.tree_generate(x[feature_values < tree.split_value], y[feature_values < tree.split_value])

            tree.leaf_num += (tree.subtree[up_part].leaf_num + tree.subtree[down_part].leaf_num)

            tree.high = max(tree.subtree[up_part].high, tree.subtree[down_part].high) + 1

        return tree

    def choose_best_split(self, x, y):
        """选择最优属性进行划分"""

        if self.critertion == 'gain':
            return choose_best_split_with_gain(x, y)
        elif self.critertion == 'gini':
            return choose_best_split_with_gini(x, y)
        elif self.critertion == 'entropy':
            return choose_best_split_with_entropy(x, y)