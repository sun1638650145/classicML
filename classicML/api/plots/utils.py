from classicML.api.plots import _plt as plt


def _set_bayes_axis(axes):
    """设置贝叶斯画布的坐标轴."""
    axes.spines['top'].set_color('none')
    axes.spines['right'].set_color('none')


def _set_lda_axis(axes):
    """设置线性判别分析画布的坐标轴."""
    axes.spines['top'].set_color('none')
    axes.spines['right'].set_color('none')


def _set_history_axis_and_background(axes):
    """设置历史记录画布的坐标轴."""
    axes.spines['left'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.spines['bottom'].set_color('none')

    axes.patch.set_facecolor('gray')
    axes.patch.set_alpha(0.1)  # 设置透明度
    axes.grid(axis='y', linestyle='dotted')


def _set_svc_axis(axes):
    """设置支持向量分类器画布的坐标轴.
    具体实现参看_set_history_axis_and_background
    """
    _set_history_axis_and_background(axes)


def _set_tree_axis(axes):
    """设置支持树画布的坐标轴."""
    axes.spines['left'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.spines['bottom'].set_color('none')

    axes.set_xticks([])
    axes.set_yticks([])


def _bayes_plot_config(x_label, y_label):
    """设置贝叶斯画布的其他配置项"""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def _history_plot_config():
    """设置历史记录画布的其他配置项.
    """
    plt.xlabel('epochs')
    plt.legend()


def _lda_plot_config(x_label, y_label):
    """设置线性判别分析画布的其他配置项"""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis([0, 1, 0, 1])
    plt.legend()


def _logistic_regression_plot_config(x_label, y_label):
    """设置逻辑回归画布的其他配置项"""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis([0, 1, 0, 1])
    plt.legend()


def _svc_plot_config(CS, kernel_name, C, x_label, y_label):
    """设置支持向量分类器的其他配置项."""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.clabel(CS, fmt={CS.levels[0]: '决策边界'})
    plt.title('kernel={}, C={}'.format(kernel_name, C))
    plt.legend()


def _add_node(parent_node, child_node, node_name, node_type, axes):
    """添加结点信息.

     Arguments:
        parent_node: tuple,
            父结点的坐标.
        child_node: tuple,
            自身结点的坐标.
        node_name: str,
            当前结点的名称文本内容.
        node_type: dict,
            结点的类型信息(不同的结点使用不同的图例).
        axes: matplotlib.axes._subplots.AxesSubplot,
            matplotlib的绘图工具.
    """
    axes.annotate(node_name,
                  xy=[parent_node[0], parent_node[1] - 0.05],
                  xytext=child_node,
                  xycoords='axes fraction',
                  textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='<-'),
                  verticalalignment='center',
                  horizontalalignment='center',
                  size=15,
                  bbox=node_type)


def _add_text(parent_node, child_node, text, axes):
    """添加文本信息.

    Arguments:
        parent_node: tuple,
            父结点的坐标.
        child_node: tuple,
            自身结点的坐标.
        text: str,
            文本内容.
        axes: matplotlib.axes._subplots.AxesSubplot,
            matplotlib的绘图工具.
    """
    x = (parent_node[0] - child_node[0]) / 2 + child_node[0]
    y = (parent_node[1] - child_node[1]) / 2 + child_node[1]

    axes.text(x, y, text, fontdict=dict(size=9))


def _tree_plot_config(criterion, pruning):
    """设置树的其他配置项."""
    if pruning:
        name = pruning.name
    else:
        name = None
    plt.title('criterion={}, pruning={}'.format(criterion, name),
              y=-0.1)