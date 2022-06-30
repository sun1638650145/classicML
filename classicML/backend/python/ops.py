"""classicML的底层核心操作."""
from typing import List, Union

import numpy as np

import classicML as cml
from classicML import _cml_precision
from classicML import CLASSICML_LOGGER

__version__ = 'backend.python.ops.0.14'


def bootstrap_sampling(x, y=None, seed=None):
    """对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    x = np.asarray(x)
    num_of_samples = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # 检查样本的第一维是否一致.
        if num_of_samples != y.shape[0]:
            raise ValueError('两个数组长度不一致[%d, %d].' % (num_of_samples, y.shape[0]))

    # 进行随机采样, 生成索引.
    np.random.seed(seed)  # 设置随机种子.
    indices = np.random.choice(a=np.arange(num_of_samples), size=num_of_samples)

    if y is not None:
        return x[indices], y[indices]
    else:
        return x[indices]


class ConvexHull(object):
    """使用Graham扫描算法计算二维凸包.

    Attributes:
        points: np.ndarray or list, 计算凸包的点.
        hull: np.ndarray, 凸包的点.

    References:
        - [Graham Scan Algorithm](https://lvngd.com/blog/convex-hull-graham-scan-algorithm-python/)
    """

    def __init__(self, points: Union[np.ndarray, List]):
        """
        Args:
            points: np.ndarray or list, 计算凸包的点.
        """
        super(ConvexHull, self).__init__()
        self.points = np.asarray(points)
        self.hull = self.compute_convex_hull()

    def compute_convex_hull(self) -> np.ndarray:
        """计算二维凸包.

        Return:
            二维凸包.
        """
        hull = []
        # 对点进行排序, 优先根据x升序.
        self.points = self.points[np.argsort(self.points[:, 0])]

        start, self.points = self.points[0], self.points[1:, ]
        hull.append(start)

        # 对点进行排序, 优先根据x斜率.
        slopes = []
        for i, point in enumerate(self.points):
            slopes.append([i, self._get_slope(start, point)])  # 逐元素计算斜率.
        slopes.sort(key=lambda x: x[1])  # 根据斜率排序.
        slopes_indices = np.asarray(slopes, dtype=cml.int32)[:, 0]

        self.points = self.points[slopes_indices]

        for point in self.points:
            hull.append(point)
            while len(hull) > 2 and self._get_cross_product(hull[-3], hull[-2], hull[-1]) < 0:
                hull.pop(-2)  # 删除序列中间的那个.

        # 增加第一个元素成为闭环.
        hull.append(hull[0])

        return np.asarray(hull)

    @staticmethod
    def _get_slope(p0: np.ndarray, p1: np.ndarray) -> float:
        """计算斜率.

        Return:
            斜率.
        """
        if p0[0] == p1[0]:
            return np.inf
        else:
            return (p0[1] - p1[1]) / (p0[0] - p1[0])

    @staticmethod
    def _get_cross_product(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算向量积.

        Return:
            向量积.
        """
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])  # x1y2 - x2y1


def calculate_centroids(x: np.ndarray,
                        clusters: np.ndarray) -> np.ndarray:
    """计算均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        clusters: numpy.ndarray, 当前的簇标记.

    Return:
        均值向量.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    n_clusters = np.unique(clusters).size

    centroids = []
    for cluster in range(n_clusters):
        centroids.append(np.mean(x[clusters == cluster], axis=0))

    return np.asarray(centroids, dtype=_cml_precision.float)


def calculate_error(x, y, i, kernel, alphas, non_zero_alphas, b):
    """计算KKT条件的违背值.

    Arguments:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        i: int, 第i个样本.
        kernel: classicML.kernel.Kernels 实例, 分类器使用的核函数.
        alphas: numpy.ndarray, 拉格朗日乘子.
        non_zero_alphas: numpy.ndarray, 非零拉格朗日乘子.
        b: float, 偏置项.

    Returns:
        KKT条件的违背值.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    x_i = x[[i], :]
    y_i = y[i, 0]

    if non_zero_alphas.any():
        # 提取全部合格的标签和对应的拉格朗日乘子.
        valid_x = x[non_zero_alphas]
        valid_y = y[non_zero_alphas]
        valid_alphas = alphas[non_zero_alphas]

        kappa = kernel(valid_x, x_i)

        fx = (np.matmul((valid_alphas.reshape(-1, 1) * valid_y).T, kappa.T) + b).astype(_cml_precision.float)
    else:
        # 拉格朗日乘子全是零的时候, 每个样本都不会对结果产生影响.
        fx = b

    error = fx - y_i

    return np.squeeze(error)


# TODO(Steve Sun, tag:code): 实现完整计算欧式距离.
def calculate_euclidean_distance(x0: np.ndarray,
                                 x1: np.ndarray) -> np.ndarray:
    """计算欧式距离.

    Args:
        x0, x1: numpy.ndarray, 要计算欧式距离的两个值.

    Return:
        欧式距离.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    distances = np.expand_dims(x0, axis=1) - x1
    distances = np.linalg.norm(distances, ord=2, axis=-1)

    return distances


def clip_alpha(alpha, low, high):
    """修剪拉格朗日乘子.

    Arguments:
        alpha: float, 拉格朗日乘子.
        low: float, 正则化系数的下界.
        high: float, 正则化系数的上界.

    Returns:
        修剪后的拉格朗日乘子.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    if alpha > high:
        alpha = high
    elif alpha < low:
        alpha = low

    return alpha


def compare_differences(x0: np.ndarray,
                        x1: np.ndarray,
                        tol: float) -> np.ndarray:
    """比较差异.

    Args:
        x0, x1: numpy.ndarray, 要比较差异的两个值.
        tol: float, 最小差异阈值.

    Return:
        差异向量.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    differences = np.abs(x0 - x1)

    return np.any(differences > tol, axis=-1)


def get_cluster(distances: np.ndarray) -> np.ndarray:
    """获得簇标记.

    Args:
        distances: numpy.ndarray, 距离.

    Return:
        簇标记.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    return np.argmin(distances, axis=-1).astype(cml.int32)


def get_conditional_probability(samples_on_attribute,
                                samples_in_category,
                                num_of_categories,
                                smoothing):
    """获取类条件概率.

    Arguments:
        samples_on_attribute: int, 在某个属性的样本.
        samples_in_category: int, 在某个类别上的样本.
        num_of_categories: int, 类别的数量.
        smoothing: bool, 是否使用平滑.

    Returns:
        类条件概率.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    if smoothing:
        samples_on_attribute += 1
        samples_in_category += num_of_categories

    probability = samples_on_attribute / samples_in_category

    return _cml_precision.float(probability)


def get_dependent_prior_probability(samples_on_attribute_in_category,
                                    number_of_sample,
                                    values_on_attribute,
                                    smoothing):
    """获取有依赖的类先验概率.

    Arguments:
        samples_on_attribute_in_category: int, 类别为c的属性i上取值为xi的样本.
        number_of_sample: int, 样本的总数.
        values_on_attribute: int, 在属性i上的取值数.
        smoothing: bool, 是否使用平滑.

    Returns:
        类先验概率.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    if smoothing:
        probability = (samples_on_attribute_in_category + 1) / (number_of_sample + 2 * values_on_attribute)  # 执行平滑操作.
    else:
        probability = samples_on_attribute_in_category / number_of_sample

    return probability


def get_prior_probability(number_of_sample, y, smoothing):
    """获取类先验概率.

    Arguments:
        number_of_sample: int, 样本的总数.
        y: numpy.ndarray, 标签.
        smoothing: bool, 是否使用平滑.

    Returns:
        类先验概率.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    if smoothing:
        p_0 = (len(y[y == 0]) + 1) / (number_of_sample + len(np.unique(y)))  # 执行平滑操作.
    else:
        p_0 = len(y[y == 0]) / number_of_sample

    return _cml_precision.float(p_0), _cml_precision.float(1 - p_0)


def get_probability_density(sample, mean, var):
    """获得概率密度.

    Arguments:
        sample: float, 样本的取值.
        mean: float, 样本在某个属性的上的均值.
        var: float, 样本在某个属性上的方差.

    Returns:
        概率密度.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    probability = 1 / (np.sqrt(2 * np.pi) * var) * np.exp(-(sample - mean) ** 2 / (2 * var ** 2))
    probability = _cml_precision.float(probability)

    if probability == 0.0:
        probability = _cml_precision.float(1e-36)  # probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.

    return probability


def get_w(S_w, mu_0, mu_1):
    """获得投影向量.

    DEPRECATED:
      `ops.get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.get_w_v2`.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    CLASSICML_LOGGER.warn('`ops.get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.get_w_v2`.')

    S_w_i = np.linalg.inv(S_w)
    w = np.matmul(S_w_i, (mu_0 - mu_1).T)

    return w.reshape(1, -1)


def get_w_v2(S_w, mu_0, mu_1):
    """获得投影向量.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.

    Notes:
        - 第二版使用奇异值分解来计算类内散度矩阵的逆矩阵.
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    U, Sigma, V_t = np.linalg.svd(S_w)
    S_w_i = V_t.T * np.linalg.inv(np.diag(Sigma)) * U.T

    w = np.matmul(S_w_i, (mu_0 - mu_1).T)

    return w.reshape(1, -1)


def get_within_class_scatter_matrix(X_0, X_1, mu_0, mu_1):
    """获得类内散度矩阵.

    Arguments:
        X_0: numpy.ndarray, 反例集合.
        X_1: numpy.ndarray, 正例集合.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        类内散度矩阵.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    S_0 = np.matmul((X_0 - mu_0).T, (X_0 - mu_0))
    S_1 = np.matmul((X_1 - mu_1).T, (X_1 - mu_1))

    S_w = S_0 + S_1

    return S_w


#TODO(Steve R. Sun, tag:code): 添加设置随机种子的功能.
def init_centroids(x: np.ndarray,
                   n_clusters: int,
                   init: Union[str, list, np.ndarray]) -> np.ndarray:
    """初始化初始均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        n_clusters: int, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

    Return:
        均值向量.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.

    Raises:
        ValueError: 聚类簇数量与初始化均值向量数量不一致, 非法索引或不能自动转换的非法均值向量.
        TypeError: 非法均值向量.
    """
    if init == 'random':
        centroid_indices = np.random.randint(low=0, high=len(x), size=n_clusters)
        centroids = x[centroid_indices, :]
    elif type(init) in (list, np.ndarray):
        init = np.squeeze(init)
        if len(init.shape) == 1:  # 指定训练数据的索引.
            if len(init) != n_clusters:
                CLASSICML_LOGGER.error('你设置聚类簇数量与初始化均值向量数量不一致[%d, %d].', n_clusters, len(init))
                raise ValueError('你设置聚类簇数量与初始化均值向量数量不一致[%d, %d].' % (n_clusters, len(init)))
            elif np.min(init) < 0 or np.max(init) >= len(x):
                CLASSICML_LOGGER.error('你使用了非法的索引(请检查索引是否越界或负值).')
                raise ValueError('你使用了非法的索引(请检查索引是否越界或负值).')

            centroids = x[init, :]
        else:  # 直接给定具体的均值向量.
            if init.shape != (n_clusters, x.shape[1]):
                CLASSICML_LOGGER.error('你使用了非法的均值向量.')
                raise ValueError('你使用了非法的均值向量.')

            centroids = np.asarray(init, dtype=_cml_precision.float)
    else:
        CLASSICML_LOGGER.error('你使用了非法的均值向量.')
        raise TypeError('你使用了非法的均值向量.')

    return centroids


def select_second_alpha(error, error_cache, non_bound_alphas):
    """选择第二个拉格朗日乘子, SMO采用的是启发式寻找的思想,
    找到目标函数变化量足够大, 即选取变量样本间隔最大.

    Arguments:
        error: float,
            KKT条件的违背值.
        error_cache: numpy.ndarray,
            KKT条件的违背值缓存.
        non_bound_alphas: numpy.ndarray,
            非边界拉格朗日乘子.

    Returns:
        拉格朗日乘子的下标和违背值.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
        - Python和CC存在精度差异, 存在潜在的可能性导致使用同样的数据和随机种子但不同后端的结果不一致,
          不过随着训练的轮数的增加, 这种差异会逐渐消失.
    """
    non_bound_index = non_bound_alphas.nonzero()[0]
    delta_e = np.abs(error - error_cache[non_bound_index])

    index_alpha = non_bound_index[np.argmax(delta_e)]  # 选取间隔最大的下标.
    error_alpha = error_cache[index_alpha]

    return index_alpha, error_alpha


def type_of_target(y):
    """判断输入数据的类型.

    Arguments:
        y: numpy.ndarray,
            待判断类型的数据.

    Returns:
        'binary': 元素只有两个离散值, 类型不限.
        'continuous': 元素都是浮点数, 且不是对应整数的浮点数.
        'multiclass': 元素不只有两个离散值, 类型不限.
        'multilabel': 元素标签不为一, 类型不限.
        'unknown': 类型未知.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
        - Python和CC针对str类型的返回暂不相同.
    """
    if y.dtype == object or np.ndim(y) > 2:
        return 'unknown'

    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        return 'continuous'

    if np.ndim(y) == 1:
        if len(np.unique(y)) == 2:
            return 'binary'
        elif len(np.unique(y)) > 2:
            return 'multiclass'
    elif y.shape[1] == 1:
        if len(np.unique(y)) == 2:
            return 'binary'
        elif len(np.unique(y)) > 2:
            return 'multiclass'

    if np.ndim(y) == 2 and len(np.unique(y)) >= 2:
        return 'multilabel'

    return 'unknown'
