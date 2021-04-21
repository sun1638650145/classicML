import numpy as np
import pandas as pd

from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _set_bayes_axis
from classicML.api.plots.utils import _bayes_plot_config
from classicML import CLASSICML_LOGGER


def _plot_background(bayes, x):
    """绘制背景分界图.

    Arguments:
        bayes: classicML.models.NB or classicML.models.SPODE,
            朴素贝叶斯分类器或超父独依赖估计器实例.
        x: numpy.ndarray, array-like, 特征数据.
    """
    # 生成向量空间.
    x0_min, x0_max = x.values[:, 0].min() - 0.1, x.values[:, 0].max() + 0.1
    x1_min, x1_max = x.values[:, 1].min() - 0.1, x.values[:, 1].max() + 0.1
    x0_coord = np.linspace(x0_min, x0_max, 300)
    x1_coord = np.linspace(x1_min, x1_max, 300)
    vector_matrix_0, vector_matrix_1 = np.meshgrid(x0_coord, x1_coord)

    # 进行预测并改变形状.
    x_test = pd.DataFrame(np.c_[vector_matrix_0.ravel(), vector_matrix_1.ravel()], columns=x.columns)
    y_pred = bayes.predict(x_test)
    y_pred = np.asarray(y_pred).reshape(vector_matrix_0.shape)

    # 进行绘制.
    plt.pcolormesh(vector_matrix_0, vector_matrix_1, y_pred, cmap='GnBu', alpha=0.75, shading='nearest')


def _plot_scatter(x, y):
    """绘制样本点.

    Arguments:
        x: pandas.core.DataFrame, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.

    Returns:
        处理后的数据标签信息.
    """
    # 去除离散值的数据.
    discrete_list = list()
    for i, datatype in enumerate(x.dtypes):
        if datatype != ('float' or 'int'):
            discrete_list.append(x.columns[i])
    x = x.drop(labels=discrete_list, axis=1)
    x = x.iloc[:, :2]  # 由于暂时只能可视化二维空间, 就只保留两个属性.

    negative_label = [y == (0 or False)][0]
    positive_label = [y == (1 or True)][0]

    plt.scatter(x.values[negative_label, 0], x.values[negative_label, 1], c='lightcoral', marker='o', label='反例')
    plt.scatter(x.values[positive_label, 0], x.values[positive_label, 1], c='c', marker='o', label='正例')

    return x.columns


def plot_bayes(bayes, x, y):
    """可视化朴素贝叶斯分类器或超父独依赖估计器的二维示意图.

    Arguments:
        bayes: classicML.models.NB or classicML.models.SPODE,
            朴素贝叶斯分类器或超父独依赖估计器实例.
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.

    Raises:
        ValueError: 模型没有训练的错误.
    """
    if bayes.is_trained is False and bayes.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有训练')
        raise ValueError('你必须先进行训练')

    ax = plt.subplot()
    _set_bayes_axis(ax)

    # 转换数据类型.
    x = pd.DataFrame(x, columns=bayes.attribute_name)

    # 绘制背景分界图.
    _plot_background(bayes, x)

    # 绘制样本点.
    x_label, y_label = _plot_scatter(x, y)

    _bayes_plot_config(x_label, y_label)
    plt.show()
