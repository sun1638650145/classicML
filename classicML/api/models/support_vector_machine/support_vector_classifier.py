import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.backend import get_kernel
from classicML.backend import get_optimizer


class SupportVectorClassifier(object):
    """支持向量分类器.

    Attributes:
        seed: int, default=None,
            随机种子.
        support: numpy.ndarray, default=None,
            分类器的支持向量下标数组.
        support_vector: numpy.ndarray, default=None,
            分类器的支持向量数组.
        support_alpha: numpy.ndarray, default=None,
            分类器的拉格朗日乘子数组.
        support_y: numpy.ndarray, default=None,
            分类器的支持向量对应的标签数组.
        b: float, default=0,
            偏置项.
        C: float, default=1.0,
            软间隔正则化系数.
        kernel: str, classicML.kernel.Kernels 实例, default='rbf'
            分类器使用的核函数.
        gamma: {'auto', 'scale'} or float, default='auto',
            在使用高斯核, sigmoid核或者多项式核时, 核函数系数,
            使用其他核函数时无效.
            - 如果gamma是'auto', gamma = 1 / 训练数据集的特征数.
            - 如果gamma是'scale', gamma = 1 / (训练数据集的特征数 * 训练数据集的方差).
        tol: float, default=1e-3,
            停止训练的最大误差值.
        epochs: int, default=-1,
            最大的训练轮数, 如果是-1则表示需要所有的样本满足条件时,
            才能停止训练, 即没有限制.
        optimizer: classicML.optimizers.Optimizer 实例
            SMO算法.
    """
    def __init__(self, seed=None):
        """初始化分类器.

        Arguments:
            seed: int, default=None,
                随机种子.
        """
        super(SupportVectorClassifier, self).__init__()
        self.seed = np.random.seed(seed)

        self.support = None
        self.support_vector = None
        self.support_alpha = None
        self.support_y = None
        self.b = 0

        self.C = None
        self.kernel = None
        self.gamma = None
        self.tol = None
        self.epochs = None
        self.optimizer = get_optimizer('SMO')

    def compile(self, C=1.0, kernel='rbf', gamma='auto', tol=1e-3):
        """编译分类器, 配置训练时使用的超参数.

        Arguments:
            C: float, default=1.0,
                软间隔正则化系数.
            kernel: str, classicML.kernel.Kernels 实例, default='rbf',
                分类器使用的核函数.
            gamma: {'auto', 'scale'} or float, default='gamma',
                在使用高斯(径向基)核, sigmoid核或者多项式核时, 核函数系数,
                使用其他核函数时无效.
            tol: float, default=1e-3,
                停止训练的最大误差值.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol

    def fit(self, x, y, epochs=1000):
        """训练分类器.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.
            epochs: int, default=1000,
                最大的训练轮数, 如果是-1则表示需要所有的样本满足条件时,
                才能停止训练, 即没有限制.

        Returns:
            SupportVectorClassifier实例.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 数值化gamma
        if self.gamma == 'auto':
            self.gamma = 1.0 / x.shape[1]
        elif self.gamma == 'scale':
            self.gamma = 1.0 / (x.shape[1] * x.var())

        # 获取核函数
        self.kernel = get_kernel(self.kernel, self.gamma)

        # 使用SMO算法优化
        (self.support, self.support_vector, self.support_alpha,
         self.support_y, self.b) = self.optimizer(x, y, self.C, self.kernel, self.tol, epochs)

        return self

    def predict(self, x):
        """使用分类器进行预测.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.

        Returns:
            SupportVectorClassifier预测的张量数组.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.support is None:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        number_of_sample = x.shape[0]
        y_pred = np.ones((number_of_sample, ), dtype=int)

        for sample in range(number_of_sample):
            kappa = self.kernel(x[sample], self.support_vector)
            fx = np.matmul((self.support_alpha.reshape(-1, 1) * self.support_y).T, kappa.T) + self.b
            if fx < 0:
                y_pred[sample] = -1

        return y_pred