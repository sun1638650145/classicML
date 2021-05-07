"""classicML中的数据预处理模块."""
import copy
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd


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


class DummyEncoder(PreProcessor):
    """Dummy编码器.

    Attributes:
        name: str, default='dummy_encoder',
            Dummy编码器名称.
        dtype: str, default='float32',
            编码后的标签的数据类型.
        class_indices: dict,
            类标签和类索引的映射字典.
    """
    def __init__(self, name='dummy_encoder', dtype='float32'):
        """
        Arguments:
            name: str, default='dummy_encoder',
                Dummy编码器名称.
            dtype: str, default='float32',
                编码后的标签的数据类型.
        """
        super(DummyEncoder, self).__init__(name=name)
        self.dtype = dtype

        self.class_indices = dict()

    def __call__(self, labels):
        """进行Dummy编码.

        Arguments:
            labels: array-like, 原始的标签.

        Returns:
            Dummy编码后的标签.
        """
        labels = np.asarray(labels)

        num_classes = np.unique(labels)
        m = len(labels)  # 样本总数.
        n = len(num_classes)  # 类别总数.

        # 构建映射字典.
        for index, value in enumerate(num_classes):
            self.class_indices.update({value: index})

        dummy_label = np.zeros(shape=[m, n], dtype=self.dtype)
        for index, label in enumerate(labels):
            dummy_label[index][self.class_indices[label]] = 1

        return dummy_label


class Imputer(PreProcessor):
    """缺失值填充器,
    连续值将填充均值, 离散值将填充众数.

    Attributes:
        name: str, default='imputer',
            缺失值填充器名称.
    """
    def __init__(self, name='imputer'):
        """
        Arguments:
            name: str, default='imputer',
                缺失值填充器名称.
        """
        super(Imputer, self).__init__(name=name)

    def __call__(self, data):
        """进行缺失值填充.

        Arguments:
            data: array-like, 输入的数据.

        Returns:
            填充后的数据.
        """
        preprocessed_data = copy.deepcopy(data)
        for column in range(data.shape[1]):
            preprocessed_data[:, column] = self._fillna(data[:, column])

        return preprocessed_data

    @staticmethod
    def _fillna(column):
        """填充数据列中的缺失值.

        Arguments:
            column: array-like, 输入的数据列.

        Returns:
            填充后的数据列.
        """
        try:
            new_column = pd.DataFrame(column, dtype='float32')
            new_column.fillna(value=np.mean(new_column.dropna().values),
                              inplace=True)
        except ValueError:
            new_column = pd.DataFrame(column)
            new_column.fillna(value=new_column.value_counts().keys()[0][0],
                              inplace=True)

        return np.squeeze(new_column.values)


class MaxMarginEncoder(PreProcessor):
    """最大化间隔编码器, 对于支持向量机的标签编码需要将编码转换为关于超平面的.

    Attributes:
        name: str, default='max_margin_encoder',
            最大化间隔编码器名称.
        dtype: str, default='float32',
            编码后的标签的数据类型.
        class_indices: dict,
            类标签和类索引的映射字典.
    """
    def __init__(self, name='max_margin_encoder', dtype='float32'):
        """
        Arguments:
            name: str, default='max_margin_encoder',
                最大化间隔编码器名称.
            dtype: str, default='float32',
                编码后的标签的数据类型.
        """
        super(MaxMarginEncoder, self).__init__(name=name)
        self.dtype = dtype

        self.class_indices = dict()

    def __call__(self, labels):
        """进行最大化间隔编码.

        Arguments:
            labels: array-like, 原始的标签.

        Returns:
            最大化间隔编码后的标签, 类标签和类索引的映射字典.
        """
        labels = np.asarray(labels)

        num_classes = np.unique(labels)
        m = len(labels)  # 样本总数.

        # 构建映射字典.
        for index, value in enumerate(num_classes):
            self.class_indices.update({value: (-1 if index == 0 else 1)})

        max_margin_label = np.zeros(m, dtype=self.dtype)
        for i in range(m):
            max_margin_label[i] = self.class_indices[labels[i]]

        return max_margin_label, self.class_indices


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
            独热编码后的标签, 类标签和类索引的映射字典.
        """
        labels = np.asarray(labels)

        num_classes = np.unique(labels)
        m = len(labels)  # 样本总数.
        n = len(num_classes)  # 类别总数.
        _class_indices = dict()

        for index, value in enumerate(num_classes):
            _class_indices.update({value: index})
            # 构建映射字典.
            _label = np.zeros(n, dtype=self.dtype)
            _label[index] = 1
            self.class_indices.update({value: _label.tolist()})

        # 进行逐元素编码操作.
        onehot_label = np.zeros([m, n], dtype=self.dtype)
        for i in np.arange(m):
            j = _class_indices[labels[i]]
            onehot_label[i, j] = 1

        return onehot_label, self.class_indices


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
