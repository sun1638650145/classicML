from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _add_node
from classicML.api.plots.utils import _add_text
from classicML.api.plots.utils import _set_tree_axis
from classicML.api.plots.utils import _tree_plot_config

NUM_OF_LEAVES = 0
HEIGHT = -1
X = 0.5
Y = 1


def _plot_decision_tree(decision_tree, parent_node, node_text, axes):
    """绘制决策树.

     Arguments:
        decision_tree: classicML.tree._TreeNode,
            决策树实例.
        parent_node: tuple,
            父结点的坐标.
        node_text: str,
            当前结点内的文本内容.
        axes: matplotlib.axes._subplots.AxesSubplot,
            matplotlib的绘图工具.
    """
    global NUM_OF_LEAVES
    global HEIGHT
    global X
    global Y

    child_node = (X + (1 + decision_tree.num_of_leaves) / (2 * NUM_OF_LEAVES), Y)
    _add_text(parent_node, child_node, node_text, axes)

    if HEIGHT == 0:
        # 决策树桩(相当于叶结点).
        _add_node(parent_node=parent_node,
                  child_node=child_node,
                  node_name=decision_tree.category,
                  node_type=dict(boxstyle='circle, pad=0.4', fc='orange'),
                  axes=axes)

        return
    else:
        # 决策的分支结点.
        _add_node(parent_node=parent_node,
                  child_node=child_node,
                  node_name=decision_tree.feature_name,
                  node_type=dict(boxstyle='round, pad=0.45', fc='lightcoral'),
                  axes=axes)

        Y -= 1 / HEIGHT

        for key in decision_tree.subtree.keys():
            if decision_tree.subtree[key].leaf:
                # 叶结点.
                X += 1 / NUM_OF_LEAVES

                _add_text(child_node, (X, Y), str(key), axes)
                _add_node(parent_node=child_node,
                          child_node=(X, Y),
                          node_name=decision_tree.subtree[key].category,
                          node_type=dict(boxstyle='circle, pad=0.4', fc='orange'),
                          axes=axes)
            else:
                _plot_decision_tree(decision_tree.subtree[key], child_node, str(key), axes)

        Y += 1 / HEIGHT


def plot_tree(tree):
    """绘制树的示意图.

    Arguments:
        tree: classicML.models.Tree, 树实例.
    """
    # 使用全局变量, 减少函数的传参并且使得递归函数更易理解.
    global NUM_OF_LEAVES
    global HEIGHT
    global X
    global Y

    decision_tree = tree.tree

    NUM_OF_LEAVES = decision_tree.num_of_leaves
    HEIGHT = decision_tree.height
    X = -0.5 / NUM_OF_LEAVES
    Y = 1

    _, ax = plt.subplots()
    _set_tree_axis(ax)

    # 绘制决策树.
    _plot_decision_tree(decision_tree, (0.5, 1), '', ax)

    _tree_plot_config(tree.criterion, tree.pruner)
    plt.show()