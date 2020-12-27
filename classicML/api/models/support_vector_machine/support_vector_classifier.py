import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.backend import get_kernel
from classicML.backend import get_optimizer
from classicML.backend import io


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
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
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
        self.is_trained = False
        self.is_loaded = False

    def compile(self, C=1.0, kernel='rbf', gamma='auto', tol=1e-3):
        """编译分类器, 配置训练时使用的超参数.

        Arguments:
            C: float, default=1.0,
                软间隔正则化系数.
            kernel: str, classicML.kernel.Kernels 实例, default='rbf',
                分类器使用的核函数.
            gamma: {'auto', 'scale'} or float, default='auto',
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

        # 标记训练完成
        self.is_trained = True

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
        if self.is_trained is False and self.is_loaded is False:
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

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            KeyError: 模型权重加载失败.

        Notes:
            模型将不会加载关于优化器的超参数.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='r',
                                                   model_name='SupportVectorClassifier')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.C = compile_ds.attrs['C']
            self.gamma = compile_ds.attrs['gamma']
            self.kernel = get_kernel(compile_ds.attrs['kernel'], self.gamma)
            self.tol = compile_ds.attrs['tol']

            self.support = weights_ds.attrs['support']
            self.support_vector = weights_ds.attrs['support_vector']
            self.support_alpha = weights_ds.attrs['support_alpha']
            self.support_y = weights_ds.attrs['support_y']
            self.b = weights_ds.attrs['b']
            # 标记加载完成
            self.is_loaded = True
        except KeyError:
            CLASSICML_LOGGER.error('模型权重加载失败, 请检查文件是否损坏')
            raise KeyError('模型权重加载失败')

    def save_weights(self, filepath):
        """将模型权重保存为一个HDF5文件.

        Arguments:
            filepath: str, 权重文件保存的路径.

        Raises:
            TypeError: 模型权重保存失败.

        Notes:
            模型将不会保存关于优化器的超参数.
        """
        # 初始化权重文件.
        parameters_gp = io.initialize_weights_file(filepath=filepath,
                                                   mode='w',
                                                   model_name='SupportVectorClassifier')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['C'] = self.C
            compile_ds.attrs['kernel'] = self.kernel.name
            compile_ds.attrs['gamma'] = self.gamma
            compile_ds.attrs['tol'] = self.tol

            weights_ds.attrs['support'] = self.support
            weights_ds.attrs['support_vector'] = self.support_vector
            weights_ds.attrs['support_alpha'] = self.support_alpha
            weights_ds.attrs['support_y'] = np.asarray(self.support_y, dtype=np.float64)
            weights_ds.attrs['b'] = np.float64(self.b)
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')