import numpy as np

from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _set_lda_axis
from classicML.api.plots.utils import _lda_plot_config
from classicML.api.plots.utils import _logistic_regression_plot_config
from classicML import CLASSICML_LOGGER


def _plot_sample_scatter(x, y):
    """绘制样本点.

    Arguments:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.

    Returns:
        反正例样本点.
    """
    negative_label = [y == 0]
    positive_label = [y == 1]
    X_0 = x[tuple(negative_label)]
    X_1 = x[tuple(positive_label)]
    plt.scatter(X_0[:, 0], X_0[:, 1], c='lightcoral', marker='o', label='反例')
    plt.scatter(X_1[:, 0], X_1[:, 1], c='c', marker='o', label='正例')

    return X_0, X_1


def plot_linear_discriminant_analysis(lda, x, y, x_label=None, y_label=None):
    """可视化线性判别分析二维示意图.

    Arguments:
        lda: classicML.models.LDA, 线性判别分析实例.
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.

    Raises:
        ValueError: 模型没有训练的错误.
    """
    if lda.is_trained is False and lda.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有权重')
        raise ValueError('你必须先进行训练')

    _, ax = plt.subplots(figsize=(5, 5))
    _set_lda_axis(ax)

    # 绘制样本点
    X_0, X_1 = _plot_sample_scatter(x, y)

    # 绘制投影向量
    x_coord = np.linspace(0, 1)
    y_coord = (lda.w[0, 1] / lda.w[0, 0]) * x_coord  # 直线经过向量w
    plt.plot(x_coord, y_coord, c='orange')

    unit_w = lda.w / np.linalg.norm(lda.w)  # 求向量w的单位向量

    # 绘制投影点(先计算对称阵)
    X_0_projecting = np.dot(X_0, np.dot(unit_w.T, unit_w))
    X_1_projecting = np.dot(X_1, np.dot(unit_w.T, unit_w))
    plt.scatter(X_0_projecting[:, 0], X_0_projecting[:, 1], c='lightcoral')
    plt.scatter(X_1_projecting[:, 0], X_1_projecting[:, 1], c='c')

    # 绘制投影线
    for i in range(X_0_projecting.shape[0]):
        plt.plot([X_0[i, 0], X_0_projecting[i, 0]], [X_0[i, 1], X_0_projecting[i, 1]], c='lightcoral', linestyle='--')
    for i in range(X_1_projecting.shape[0]):
        plt.plot([X_1[i, 0], X_1_projecting[i, 0]], [X_1[i, 1], X_1_projecting[i, 1]], c='c', linestyle='--')

    # 绘制样本中心
    X_0_center = np.dot(lda.mu_0, np.dot(unit_w.T, unit_w))
    X_1_center = np.dot(lda.mu_1, np.dot(unit_w.T, unit_w))
    plt.scatter(X_0_center[:, 0], X_0_center[:, 1], s=60, c='red', marker='h', label='反例样本中心')
    plt.scatter(X_1_center[:, 0], X_1_center[:, 1], s=60, c='mediumblue', marker='h', label='正例样本中心')

    _lda_plot_config(x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()


def plot_logistic_regression(logistic_regression, x, y, x_label=None, y_label=None):
    """可视化逻辑回归二维示意图.

    Arguments:
        logistic_regression: classicML.models.LogisticRegression, 逻辑回归实例.
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.

    Raises:
        ValueError: 模型没有训练的错误.
    """
    if logistic_regression.is_trained is False and logistic_regression.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有训练')
        raise ValueError('你必须先进行训练')

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[1] != 2:
        CLASSICML_LOGGER.error('x的维度异常')
        raise ValueError('无法可视化')

    # 绘制样本点
    _plot_sample_scatter(x, y)

    # 绘制回归方程
    x_coord = np.linspace(0, 1)
    # 0 = x1 * beta[0] + x2 * beta[1] + beta[2]
    y_coord = -(logistic_regression.beta[0] * x_coord + logistic_regression.beta[2]) / logistic_regression.beta[1]
    plt.plot(x_coord, y_coord, c='orange', label='回归方程-logistic regression')

    _logistic_regression_plot_config(x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()
