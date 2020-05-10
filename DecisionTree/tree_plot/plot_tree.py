import matplotlib.pyplot as plt
# 设置中文
plt.rcParams["font.family"] = 'Arial Unicode MS'


decision_node = dict(boxstyle='round, pad=0.45', fc='darkturquoise')# 划分点框样式和颜色
leaf_node = dict(boxstyle='circle, pad=0.4', fc='skyblue')# 叶结点框样式和颜色
arrow_args = dict(arrowstyle="<-")# 箭头的样式


x_coord = None
y_coord = None
num_of_leaf = None
high_of_tree = None


def plot_text(mid_text, center_point, parent_point, ax):
    """
    填写特征标签

        Parameters
        ----------
        mid_text : 特征名称

        center_point : 指向结点在图上的坐标

        parent_point : 前导结点的坐标

        ax: 绘图对象
    """
    x_mid = (parent_point[0] - center_point[0]) / 2 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2 + center_point[1]

    ax.text(x_mid, y_mid, mid_text, fontdict=dict(size=9))


def plot_node(node_name, center_point, parent_point, node_type, ax):
    """
    绘制结点和指向线

        Parameters
        ----------
        node_name : 结点的名称, 如果是叶结点实际上就是结点所属的种类(例如:好坏); 非叶结点实际上是划分的属性名称(例如:黑白灰)

        center_point : 结点在图上的坐标

        parent_point : 前导结点的坐标

        node_type : 结点的类型(叶结点或者划分结点)

        ax : 绘图对象
    """
    ax.annotate(node_name,# 显示的文本
                xy=[parent_point[0], parent_point[1] - 0.05],
                xytext=center_point,# 结点在图上的坐标
                xycoords='axes fraction',# xy放置位置的参考（左下角的轴）
                textcoords='axes fraction',# xytext放置位置的参考（左下角的轴）默认与xycroods相同
                arrowprops=arrow_args,# 注释的箭头样式
                verticalalignment='center',# xy垂直对齐方式
                horizontalalignment='center',# xy水平对齐方式
                size=15,# xy文本字号
                bbox=node_type)# 在文本周围绘制框 结点类型



def plot_engine(tree, parent_point, node_text, ax):
    """可视化决策树"""
    global x_coord
    global y_coord
    global num_of_leaf
    global high_of_tree

    leaf_num = tree.leaf_num
    center_point = (x_coord + (1 + leaf_num) / (2 * num_of_leaf), y_coord)

    plot_text(node_text, center_point, parent_point, ax)

    if high_of_tree == 0:
        # 如果只有一个根结点或者到了叶结点
        plot_node(tree.leaf_class, center_point, parent_point, leaf_node, ax)
        return
    else:
        # 多个结点划分点就是属性名称
        plot_node(tree.feature_name, center_point, parent_point, decision_node, ax)

        # 改变当前点的y坐标, 子树绘制完成后修改回原坐标
        y_coord -= 1 / high_of_tree

        for key in tree.subtree.keys():
            if tree.subtree[key].is_leaf:
                x_coord += 1 / num_of_leaf
                # 叶结点
                plot_node(tree.subtree[key].leaf_class, (x_coord, y_coord), center_point, leaf_node, ax)
                plot_text(str(key), (x_coord, y_coord), center_point, ax)
            else:
                plot_engine(tree.subtree[key], center_point, str(key), ax)

        y_coord += 1 / high_of_tree


def plot_decision_tree(tree):
    """绘制决策树"""
    global x_coord
    global y_coord
    global num_of_leaf
    global high_of_tree

    num_of_leaf = tree.leaf_num
    high_of_tree = tree.high
    # 确定初始位置(第一象限)
    x_coord = -0.5 / num_of_leaf
    y_coord = 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.6)

    # 隐藏坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 隐藏坐标轴
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plot_engine(tree, (0.5, 1), '', ax)
    plt.show()
