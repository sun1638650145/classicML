from pathlib import Path
from typing import List, Union

import numpy as np

from classicML import _cml_precision
from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import init_centroids
from classicML.backend import calculate_euclidean_distance
from classicML.backend import get_cluster
from classicML.backend import calculate_centroids
from classicML.backend import compare_differences
from classicML.backend import io


class KMeans(BaseModel):
    """K-均值聚类.

    Attributes:
        n_clusters: int, default=3, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, default='random', 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.
        tol: float, default=1e-3,
            停止训练的最小调整幅度阈值.
        centroids: numpy.ndarray, 均值向量(簇质心).
        clusters: numpy.ndarray, 经过训练后数据的簇标记.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self, n_clusters: int = 3):
        """初始化K-均值聚类.

        Args:
            n_clusters: int, default=3, 聚类簇的数量.
        """
        super(KMeans, self).__init__()

        self.n_clusters = n_clusters

        self.init = None
        self.tol = -1
        self.centroids = None
        self.clusters = None
        self.is_trained = False
        self.is_loaded = False

    def compile(self,
                init: Union[str, List, np.ndarray] = 'random',
                tol: float = 1e-3,
                **kwargs):
        """编译K-均值聚类.

        Args:
            init: 'random', list or numpy.ndarray, default='random', 均值向量的初始化方式,
                'random': 随机初始化;
                list or numpy.ndarray: 可以指定训练数据的索引或者值, 也可以直接给定具体的均值向量.
            tol: float, default=1e-3,
                停止训练的最小调整幅度阈值.
        """
        self.init = init
        self.tol = tol

    def fit(self,
            x: np.ndarray,
            epochs: int = 100,
            **kwargs) -> 'KMeans':
        """训练K-均值聚类.

        Args:
            x: numpy.ndarray, 特征数据.
            epochs: int, default=100,
                最大的训练轮数, 如果均值向量已经不更新将会提前自动结束训练.

        Return:
            K-means实例.
        """
        x = np.asarray(x, dtype=_cml_precision.float)

        # 初始化初始均值向量.
        self.centroids = init_centroids(x, self.n_clusters, self.init)

        for _ in range(epochs):
            # 计算样本和各均值向量的距离.
            distances = calculate_euclidean_distance(x, self.centroids)
            # 划入相应的簇.
            self.clusters = get_cluster(distances)
            # 计算新均值向量.
            new_centroids = calculate_centroids(x, self.clusters)
            # 更新均值向量.
            if compare_differences(self.centroids, new_centroids, self.tol).any():
                self.centroids = new_centroids
            else:  # 均值向量均未更新, 则提前退出.
                break

        self.is_trained = True

        return self

    def predict(self,
                x: Union[np.ndarray, List],
                **kwargs) -> np.ndarray:
        """使用K-均值聚类预测新样本所在的簇.

        Args:
            x: numpy.ndarray or list, 特征数据.

        Return:
            新样本所在的簇.

        Raise:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        x = np.asarray(x, dtype=_cml_precision.float)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        # 计算样本和各均值向量之间的距离.
        distances = calculate_euclidean_distance(x, self.centroids)
        # 划入相应的簇.
        clusters = get_cluster(distances)

        return clusters

    def score(self, x, y=None):
        """"
        Raise:
            NotImplementedError: 无监督学习(聚类)没有score方法.
        """
        CLASSICML_LOGGER.error('无监督学习(聚类)没有score方法')
        raise NotImplementedError('无监督学习(聚类)没有score方法')

    def load_weights(self, filepath: Union[str, Path]):
        """加载模型参数.

        Args:
            filepath: str, 权重文件加载的路径.

        Raise:
            KeyError: 模型权重加载失败.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='r',
                                                   model_name='KMeans')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.init = compile_ds.attrs['init']
            self.tol = compile_ds.attrs['tol']

            self.centroids = weights_ds.attrs['centroids']
            self.clusters = weights_ds.attrs['clusters']
            # 标记加载完成
            self.is_loaded = True
        except KeyError:
            CLASSICML_LOGGER.error('模型权重加载失败, 请检查文件是否损坏')
            raise KeyError('模型权重加载失败')

    def save_weights(self, filepath: Union[str, Path]):
        """将模型权重保存为一个HDF5文件.

        Args:
            filepath: str, 权重文件保存的路径.

        Raise:
            TypeError: 模型权重保存失败.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='w',
                                                   model_name='KMeans')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['init'] = self.init
            compile_ds.attrs['tol'] = self.tol

            weights_ds.attrs['centroids'] = self.centroids
            weights_ds.attrs['clusters'] = self.clusters
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')
