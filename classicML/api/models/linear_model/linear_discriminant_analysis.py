import re
from time import time

import h5py
import numpy as np

from classicML import CLASSICML_LOGGER
from classicML.backend import get_within_class_scatter_matrix
from classicML.backend import get_w


class LinearDiscriminantAnalysis(object):
    """线性判别分析.

    Attributes:
        w: numpy.ndarray, 投影向量.
        mu_0: numpy.ndarray, 反例的均值.
        mu_1: numpy.ndarray, 正例的均值.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self):
        super(LinearDiscriminantAnalysis, self).__init__()
        self.w = None
        self.mu_0 = None
        self.mu_1 = None
        self.is_trained = False
        self.is_loaded = False

    def fit(self, x, y):
        """训练模型.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.
            y: numpy.ndarray, array-like, 标签.

        Returns:
            LDA实例.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        X_0 = x[y == 0]
        X_1 = x[y == 1]

        self.mu_0 = np.mean(X_0, axis=0, keepdims=True)
        self.mu_1 = np.mean(X_1, axis=0, keepdims=True)

        # 获得类内散度矩阵
        S_w = get_within_class_scatter_matrix(X_0, X_1, self.mu_0, self.mu_1)
        # 获得投影向量
        self.w = get_w(S_w, self.mu_0, self.mu_1)
        # 标记训练完成
        self.is_trained = True

        return self

    def predict(self, x):
        """模型进行预测.

        Arguments:
            x: numpy.ndarray, array-like, 特征数据.

        Returns:
            LDA预测的标签张量.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有权重')
            raise ValueError('你必须先进行训练')

        coord = np.dot(x, self.w.T)

        center_0 = np.dot(self.w, self.mu_0.T)
        center_1 = np.dot(self.w, self.mu_1.T)

        y_pred = np.abs(coord - center_0) > np.abs(coord - center_1)
        y_pred = np.squeeze(y_pred)
        y_pred = y_pred.astype(int)

        return y_pred

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            IOError: 文件打开失败.
            ValueError: 文件核验失败.
            KeyError: 权重加载失败.
        """
        try:
            fp = h5py.File(filepath, 'r')
        except IOError:
            CLASSICML_LOGGER.error('文件打开失败, 请检查文件是否存在或损坏')
            raise IOError('文件打开失败')

        description_gp = fp['description']
        _version = description_gp.attrs['__version__']
        _model_name = description_gp.attrs['model_name']

        if _model_name != 'LinearDiscriminantAnalysis':
            CLASSICML_LOGGER.error('文件核验失败, 模型不匹配')
            raise ValueError('文件核验失败')
        # _version_list[0]是主版本号, _version_list[1]是权重文件的版本号.
        _version_list = re.findall('\\d+.?\\d*', _version)
        if _version_list[0] != '0.5' or _version_list[1] != '1':
            CLASSICML_LOGGER.error('文件核验失败, 版本不兼容')
            raise ValueError('文件核验失败')

        try:
            parameters_ds = fp['parameters']
            self.w = parameters_ds.attrs['w']
            self.mu_0 = parameters_ds.attrs['mu_0']
            self.mu_1 = parameters_ds.attrs['mu_1']
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
            IOError: 模型权重保存失败.
        """
        try:
            fp = h5py.File(filepath, 'w')
            # 创建描述信息组.
            description_gp = fp.create_group(name='description')
            description_gp.attrs['__version__'] = '0.5_weights.V1'
            description_gp.attrs['model_name'] = 'LinearDiscriminantAnalysis'
            description_gp.attrs['saved_time'] = time()
            # 创建参数数据集.
            parameters_ds = fp.create_dataset(name='parameters', dtype=np.float64)
            parameters_ds.attrs['w'] = self.w
            parameters_ds.attrs['mu_0'] = self.mu_0
            parameters_ds.attrs['mu_1'] = self.mu_1

            fp.close()
        except IOError:
            CLASSICML_LOGGER.error('模型权重保存失败.')
            raise IOError('模型权重保存失败')