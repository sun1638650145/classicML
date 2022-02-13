import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _adaboost_plot_config


def _plot_sample_scatter(x, y):
    """绘制样本点.

    Args:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.

    """
    negative_label = [y == -1][0]
    positive_label = [y == 1][0]

    plt.scatter(x[negative_label, 0], x[negative_label, 1], c='lightcoral', marker='o', label='反例')
    plt.scatter(x[positive_label, 0], x[positive_label, 1], c='c', marker='o', label='正例')


def _plot_decision_boundary(adaboost, plot_estimators):
    """绘制决策边界.

    Args:
        adaboost: classicML.models.AdaBoostClassifier,
            AdaBoost分类器实例.
        plot_estimators: bool,
            是否绘制基学习器的决策边界.

    Return:
        matplotlib的轮廓集.
    """
    # 生成向量空间.
    x0_coord = np.linspace(0, 1, 300)
    x1_coord = np.linspace(0, 1, 300)
    vector_matrix_0, vector_matrix_1 = np.meshgrid(x0_coord, x1_coord)

    # 绘制基学习器的决策边界.
    if plot_estimators:
        for estimator in adaboost.estimators:
            y_pred = estimator.predict(np.c_[vector_matrix_0.ravel(), vector_matrix_1.ravel()])
            y_pred = y_pred.reshape(vector_matrix_0.shape)
            plt.contour(vector_matrix_0, vector_matrix_1, y_pred, [0],
                        colors='gray',
                        linewidths=0.8,
                        linestyles='dashed')

    # 进行预测并改变形状.
    y_pred = adaboost.predict(np.c_[vector_matrix_0.ravel(), vector_matrix_1.ravel()])
    y_pred = y_pred.reshape(vector_matrix_0.shape)

    # 使用向量空间的点进行绘制决策边界(y_pred是到平面的高度, 代替直接绘制y=0).
    CS = plt.contour(vector_matrix_0, vector_matrix_1, y_pred, [0], colors='orange', linewidths=1.2)

    return CS


def plot_adaboost(adaboost, x, y, x_label=None, y_label=None, plot_estimators=False):
    """可视化AdaBoost分类器二维示意图.

    Args:
        adaboost: classicML.models.AdaBoostClassifier,
            AdaBoost分类器实例.
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.
        plot_estimators: bool, default=False,
            是否绘制基学习器的决策边界.

    Raise:
        ValueError: 模型没有训练的错误.
    """
    if adaboost.is_trained is False:
        CLASSICML_LOGGER.error('模型没有训练')
        raise ValueError('你必须先进行训练')

    # 绘制样本点.
    _plot_sample_scatter(x, y)

    # 绘制决策边界.
    CS = _plot_decision_boundary(adaboost, plot_estimators)

    _adaboost_plot_config(CS, x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()
