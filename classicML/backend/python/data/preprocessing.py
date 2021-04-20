from abc import ABC
from abc import abstractmethod

import numpy as np


class PreProcessor(ABC):
    """预处理器基类,
    预处理器将实现一系列预处理操作, 部分预处理器还有对应的逆操作.

    Attributes:
        name: str, default='preprocessor',
            预处理器名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
       NotImplemented: inverse方法需要用户实现.
    """
    def __init__(self, name='preprocessor'):
        """
        Arguments:
            name: str, default='preprocessor',
                预处理器名称.
        """
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """预处理操作."""
        raise NotImplementedError

    def inverse(self, *args, **kwargs):
        """预处理逆操作."""
        raise NotImplemented


class OneHotEncoder(PreProcessor):
    """独热编码器.

    Attributes:
        name: str, default='one-hot_encoder',
            独热编码器名称.
        dtype: str, default='float32',
            编码后的标签的数据类型.
        class_indices: dict,
            类标签和类索引的映射字典.
    """
    def __init__(self, name='one-hot_encoder', dtype='float32'):
        """
        Arguments:
            name: str, default='one-hot_encoder',
                独热编码器的名称.
            dtype: str, default='float32',
                编码后的标签的数据类型.
        """
        super(OneHotEncoder, self).__init__(name=name)
        self.dtype = dtype

        self.class_indices = dict()

    def __call__(self, labels):
        """进行独热编码.

        Arguments:
            labels: array-like, 原始的标签.

        Returns:
            独热编码后的标签.
        """
        labels = np.asarray(labels)

        num_classes = np.unique(labels)
        m = len(labels)  # 样本总数.
        n = len(num_classes)  # 类别总数.

        # 构建映射字典.
        for index, value in enumerate(num_classes):
            self.class_indices.update({value: index})

        onehot_label = np.zeros([m, n], dtype=self.dtype)
        for i in np.arange(m):
            j = self.class_indices[labels[i]]
            onehot_label[i, j] = 1

        return onehot_label


class StandardScaler(PreProcessor):
    """标准化器.

    Attributes:
        name: str, default='standard_scalar',
            标准化器的名称.
        dtype: str, default='float32',
            标准化后数据元素的数据类型.
        axis: int, default=-1,
            标准化所沿轴.
        mean: float, default=None,
            数据的均值.
        var: float, default=None,
            数据的方差.
    """
    def __init__(self, name='standard_scalar', dtype='float32', axis=-1):
        """
        Arguments:
            name: str, default='standard_scalar',
                标准化器的名称.
            dtype: str, default='float32',
                标准化后数据元素的数据类型.
            axis: int, default=-1,
                标准化所沿轴.
        """
        super(StandardScaler, self).__init__(name=name)
        self.dtype = dtype
        self.axis = axis

        self.mean = None
        self.var = None

    def __call__(self, data):
        """进行标准化.

        Arguments:
            data: array-like, 输入的数据.

        Returns:
            标准化后的数据.
        """
        data = np.asarray(data, dtype=self.dtype)
        self.mean = np.mean(data, axis=self.axis)
        self.var = np.var(data, axis=self.axis)

        preprocessed_data = (data - self.mean) / self.var

        return preprocessed_data.astype(self.dtype)

    def inverse(self, preprocessed_data):
        """进行反标准化.

        Arguments:
            preprocessed_data: array-like, 输入的标准化后数据.

        Returns:
            标准化前的数据.
        """
        preprocessed_data = np.asarray(preprocessed_data, dtype=self.dtype)
        data = preprocessed_data * self.var + self.mean

        return data.astype(self.dtype)


class MinMaxScaler(PreProcessor):
    """归一化器.

    Attributes:
        name: str, default='minmax_scalar',
            归一化器的名称.
        dtype: str, default='float32',
            归一化后数据元素的数据类型.
        axis: int, default=-1,
            归一化所沿轴.
        min: float, default=None,
            数据的最小值.
        max: float, default=None,
            数据的最大值.
    """
    def __init__(self, name='minmax_scalar', dtype='float32', axis=-1):
        """
        Arguments:
            name: str, default='minmax_scalar',
                归一化器的名称.
            dtype: str, default='float32',
                归一化后数据元素的数据类型.
            axis: int, default=-1,
                归一化所沿轴.
        """
        super(MinMaxScaler, self).__init__(name=name)
        self.dtype = dtype
        self.axis = axis

        self.min = None
        self.max = None

    def __call__(self, data):
        """进行归一化.

        Arguments:
            data: array-like, 输入的数据.

        Returns:
            归一化后的数据.
        """
        data = np.asarray(data, dtype=self.dtype)
        self.min = np.min(data, axis=self.axis)
        self.max = np.max(data, axis=self.axis)

        preprocessed_data = (data - self.min) / (self.max - self.min)

        return preprocessed_data.astype(self.dtype)

    def inverse(self, preprocessed_data):
        """进行反归一化.

        Arguments:
            preprocessed_data: array-like, 输入的归一化后数据.

        Returns:
            归一化前的数据.
        """
        preprocessed_data = np.asarray(preprocessed_data, dtype=self.dtype)
        data = preprocessed_data * (self.max - self.min) + self.min

        return data.astype(self.dtype)
