import numpy as np

from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _set_svc_axis
from classicML.api.plots.utils import _svc_plot_config
from classicML import CLASSICML_LOGGER


def _plot_sample_and_support_scatter(x, y, support):
    """绘制样本点和支持向量.

    Arguments:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        support: numpy.ndarray, 分类器的支持向量下标数组.
    """
    negative_label = [y == -1][0]
    positive_label = [y == 1][0]

    plt.scatter(x[negative_label, 0], x[negative_label, 1], c='lightcoral', marker='o', label='反例')
    plt.scatter(x[positive_label, 0], x[positive_label, 1], c='c', marker='o', label='正例')
    plt.scatter(x[support, 0], x[support, 1], c='white', marker='o', edgecolors='green', s=150, alpha=0.3)


def _plot_decision_boundary(svc):
    """绘制决策边界.

    Arguments:
        svc: classicML.models.SVC, 支持向量分类器实例.

    Returns:
        matplotlib的轮廓集.
    """
    # 生成向量空间.
    x0_coord = np.linspace(0, 1, 300)
    x1_coord = np.linspace(0, 0.8, 300)
    vector_matrix_0, vector_matrix_1 = np.meshgrid(x0_coord, x1_coord)

    # 进行预测并改变形状.
    y_pred = svc.predict(np.c_[vector_matrix_0.ravel(), vector_matrix_1.ravel()])
    y_pred = y_pred.reshape(vector_matrix_0.shape)

    # 使用向量空间的点进行绘制决策边界(y_pred是到平面的高度, 代替直接绘制y=0)
    CS = plt.contour(vector_matrix_0, vector_matrix_1, y_pred, [0], colors='orange', linewidths=1)

    return CS


def plot_support_vector_classifier(svc, x, y, x_label=None, y_label=None):
    """可视化支持向量分类器二维示意图.

    Arguments:
        svc: classicML.models.SVC, 支持向量分类器实例.
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.

    Raises:
        ValueError: 模型没有训练的错误.
    """
    if svc.is_trained is False and svc.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有训练')
        raise ValueError('你必须先进行训练')

    ax = plt.subplot()
    _set_svc_axis(ax)

    # 绘制样本点和支持向量
    _plot_sample_and_support_scatter(x, y, svc.support)

    # 绘制决策边界
    CS = _plot_decision_boundary(svc)

    _svc_plot_config(CS, svc.kernel.name, svc.C, x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()
