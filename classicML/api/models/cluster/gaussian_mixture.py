from typing import List, Union

import numpy as np

from classicML import _cml_precision
from classicML.api.models import BaseModel
from classicML.backend import init_centroids as init_means
from classicML.backend import init_covariances
from classicML.backend import init_mixture_coefficient
from classicML.backend import get_gaussian_mixture_distribution_posterior_probability
from classicML.backend import calculate_means
from classicML.backend import calculate_covariances
from classicML.backend import calculate_mixture_coefficient
from classicML.backend import compare_differences
from classicML.backend import get_cluster


class GaussianMixture(BaseModel):
    """高斯混合聚类.

    Attributes:
        n_components: int, default=3, 高斯混合成分的个数.
        init: 'random', list or numpy.ndarray, default='random', 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.
        covariances_init: list or numpy.ndarray, default=None, 协方差矩阵的初始化方式,
            默认将初始化为主对角线为0.1的对角阵张量, 也可以直接给定具体的协方差矩阵.
        mixture_coefficient_init: list or numpy.ndarray, default=None, 混合系数的初始化方式,
            默认将初始化为高斯混合成分的个数的倒数, 也可以直接给定具体的混合系数;
            无论任何初始化形式, 请保证混合系数的和为1.
        tol: float, default=1e-3,
            停止训练的最小调整幅度阈值.
        mu: numpy.ndarray, 均值向量.
        sigma: numpy.ndarray, 协方差矩阵.
        alpha: numpy.ndarray, 混合系数(混合系数的和为1).
        clusters: numpy.ndarray, 经过训练后数据的簇标记.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """

    def __init__(self, n_components: int = 3):
        """初始化高斯混合聚类.

        Args:
            n_components: int, default=3, 高斯混合成分的个数.
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components

        self.init = None
        self.covariances_init = None
        self.mixture_coefficient_init = None
        self.tol = -1
        self.mu = None
        self.sigma = None
        self.alpha = None
        self.clusters = None
        self.is_trained = False
        self.is_loaded = False

    def compile(self,
                init: Union[str, List, np.ndarray] = 'random',
                covariances_init: Union[List, np.ndarray, None] = None,
                mixture_coefficient_init: Union[List, np.ndarray, None] = None,
                tol: float = 1e-3,
                **kwargs):
        """编译高斯混合聚类.

        Args:
            init: 'random', list or numpy.ndarray, default='random', 均值向量的初始化方式,
                'random': 采用随机初始化;
                list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.
            covariances_init: list or numpy.ndarray, default=None, 协方差矩阵的初始化方式,
                默认将初始化为主对角线为0.1的对角阵张量, 也可以直接给定具体的协方差矩阵.
            mixture_coefficient_init: list or numpy.ndarray, default=None, 混合系数的初始化方式,
                默认将初始化为高斯混合成分的个数的倒数, 也可以直接给定具体的混合系数;
                无论任何初始化形式, 请保证混合系数的和为1.
            tol: float, default=1e-3,
                停止训练的最小调整幅度阈值.
        """
        self.init = init
        self.covariances_init = covariances_init
        self.mixture_coefficient_init = mixture_coefficient_init
        self.tol = tol

    def fit(self,
            x: np.ndarray,
            epochs: int = 100,
            **kwargs) -> 'GaussianMixture':
        """训练高斯混合聚类.

        Args:
            x: numpy.ndarray, 特征数据.
            epochs: int, default=100,
                最大的训练轮数, 如果均值向量已经不更新将会提前自动结束训练.

        Return:
            GaussianMixture实例.
        """
        x = np.asarray(x, dtype=_cml_precision.float)

        # 初始化高斯混合分布的模型参数.
        self.mu = init_means(x, self.n_components, self.init)
        self.sigma = init_covariances(x, self.n_components, self.covariances_init)
        self.alpha = init_mixture_coefficient(self.n_components, self.mixture_coefficient_init)

        for _ in range(epochs):
            # 计算各混合成分生成的后验概率(EM算法的E步).
            gamma = get_gaussian_mixture_distribution_posterior_probability(sample=x,
                                                                            mean=self.mu,
                                                                            var=self.sigma,
                                                                            alpha=self.alpha,
                                                                            n_components=self.n_components)
            # 计算新参数.
            mu = calculate_means(x, gamma)
            sigma = calculate_covariances(x, self.mu, gamma)
            alpha = calculate_mixture_coefficient(x.shape[0], gamma)
            # 更新模型参数(EM算法的M步).
            if (compare_differences(self.mu, mu, self.tol).any()
                or compare_differences(self.sigma, sigma, self.tol).any()
                    or compare_differences(self.alpha, alpha, self.tol).any()):
                self.mu = mu
                self.sigma = sigma
                self.alpha = alpha
            else:  # 均值向量均未更新, 则提前退出.
                break

        # 划入相应的簇.
        # 使用负概率是因为代码共用, KMeans: argmin(-gamma) = Gaussian: argmax(gamma).
        self.clusters = get_cluster(-gamma)

        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        pass
