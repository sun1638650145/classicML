from typing import Optional

import numpy as np

from classicML import models
from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _k_means_config
from classicML.backend import ConvexHull
from classicML import CLASSICML_LOGGER


def _plot_sample_scatter_and_centroids(x: np.ndarray,
                                       centroids: np.ndarray,
                                       clusters: Optional[np.ndarray] = None,
                                       style: str = 'k'):
    """绘制样本点和均值向量(簇质心).

    Args:
        x: numpy.ndarray, 特征数据.
        centroids: numpy.ndarray, 均值向量(簇质心).
        clusters: numpy.ndarray, default=None,
            经过训练后数据的簇标记, 使用K-均值聚类绘制风格时这个参数将无效.
        style: {'k', 'g'}, default='k',
            'k': K-均值聚类绘制风格,
            'g': 高斯混合聚类绘制风格.
    """
    if style == 'k':
        plt.scatter(x[:, 0], x[:, 1], c='lightcoral', marker='o', label='样本点')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='c', marker='o', label='均值向量(簇质心)')
    elif style == 'g':
        for centroid in range(len(centroids)):
            # 根据索引分组.
            X = x[clusters == centroid]
            plt.scatter(X[:, 0], X[:, 1], marker='o', label=f'簇{centroid}样本点')

        plt.scatter(centroids[:, 0], centroids[:, 1], c='c', marker='x', label='均值向量(簇质心)')


def plot_gaussian_mixture(gaussian_mixture: models.GaussianMixture,
                          x: np.ndarray,
                          x_label: Optional[str] = None,
                          y_label: Optional[str] = None):
    """可视化高斯混合聚类二维示意图.

    Args:
        gaussian_mixture: classicML.models.GaussianMixture, 高斯混合聚类实例.
        x: numpy.ndarray, 特征数据.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.

    Raise:
        ValueError: 模型没有训练的错误.
    """
    if gaussian_mixture.is_trained is False and gaussian_mixture.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有权重')
        raise ValueError('你必须先进行训练')

    # 绘制样本点和均值向量(簇质心).
    _plot_sample_scatter_and_centroids(x, gaussian_mixture.mu, gaussian_mixture.clusters, style='g')

    _k_means_config(x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()


def plot_k_means(k_means: models.KMeans,
                 x: np.ndarray,
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None):
    """可视化K-均值聚类二维示意图.

    Args:
        k_means: classicML.models.KMeans, K-均值聚类实例.
        x: numpy.ndarray, 特征数据.
        x_label: str, default=None,
            横轴的标签.
        y_label: str, default=None,
            纵轴的标签.

    Raise:
        ValueError: 模型没有训练的错误.
    """
    if k_means.is_trained is False and k_means.is_loaded is False:
        CLASSICML_LOGGER.error('模型没有权重')
        raise ValueError('你必须先进行训练')

    # 绘制样本点和均值向量(簇质心).
    _plot_sample_scatter_and_centroids(x, k_means.centroids, style='k')

    # 绘制每个簇的凸包.
    for cluster in range(k_means.n_clusters):
        X = x[k_means.clusters == cluster]

        hull = ConvexHull(X).hull
        plt.plot(hull[:, 0], hull[:, 1], 'c--')

    _k_means_config(x_label, y_label)

    # 自动调整子图.
    plt.tight_layout()
    plt.show()
