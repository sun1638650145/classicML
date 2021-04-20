import numpy as np
import pandas as pd

from classicML.backend.python.data.preprocessing import Imputer
from classicML.backend.python.data.preprocessing import MinMaxScaler
from classicML.backend.python.data.preprocessing import OneHotEncoder
from classicML.backend.python.data.preprocessing import StandardScaler


class Dataset(object):
    """数据集,
    数据集提供了对输入数据的预处理和封装的功能, 使之满足cml模型输入的需要.

    Attributes:
        dataset_type: {'train', 'validation', 'test'}, default='train',
            数据集的类型, 如果声明为测试集, 将不会生成对应的标签.
        label_mode: {'one-hot'}, default=None,
            标签的编码格式.
        fillna: bool, default=True,
            是否填充缺失值.
        standardization: bool, default=False,
            是否使用标准化.
        normalization: bool, default=False,
            是否使用归一化.
        name: str, default=None,
            数据集的名称.
        x: numpy.ndarray,
            处理后的特征数据.
        y: numpy.ndarray,
            处理后的标签.
        class_indices: dict,
            类标签和类索引的映射字典.
    """

    def __init__(self,
                 dataset_type='train',
                 label_mode=None,
                 fillna=True,
                 standardization=False,
                 normalization=False,
                 name=None):
        """初始化数据集.

        Arguments:
            dataset_type: {'train', 'validation', 'test'}, default='train',
                数据集的类型, 如果声明为测试集, 将不会生成对应的标签.
            label_mode: {'one-hot'}, default=None,
                标签的编码格式.
            fillna: bool, default=True,
                是否填充缺失值.
            standardization: bool, default=False,
                是否使用标准化.
            normalization: bool, default=False,
                是否使用归一化.
            name: str, default=None,
                数据集的名称.
        """
        super(Dataset, self).__init__()
        self.dataset_type = dataset_type.lower()
        self.label_mode = label_mode
        self.fillna = fillna
        self.standardization = standardization
        self.normalization = normalization
        self.name = name

        self.x = None
        self.y = None
        self.class_indices = dict()

    def from_csv(self, filepath):
        """从CSV文件中加载数据集.

        Arguments:
            filepath: str, CSV文件的路径.

        Returns:
            经过预处理的特征数据和标签.
        """
        data = pd.read_csv(filepath_or_buffer=filepath,
                           index_col=0,
                           header=0)

        # 预处理特征数据.
        self._preprocessing_features(data.iloc[:, :-1].values)

        # 预处理标签.
        if self.dataset_type != 'test':
            if len(np.unique(data.iloc[:, -1].values)) == 2:
                self._preprocessing_binary_labels(data.iloc[:, -1].values)
            else:
                self._preprocessing_categorical_babels(data.iloc[:, -1].values)
            # 编码标签.
            if self.label_mode == 'one-hot':
                self.y = OneHotEncoder()(self.y)

        return self.x, self.y

    def _preprocessing_features(self, features):
        """预处理特征值.

        Arguments:
            features: numpy.ndarray, 特征数据.
        """
        # 处理缺失值.
        if self.fillna:
            features = Imputer()(features)

        for column in range(features.shape[1]):
            # 连续值处理.
            if features[:, column].dtype in ('float32', 'float64', 'int32', 'int64'):
                if self.standardization:
                    features[:, column] = StandardScaler(axis=0)(features[:, column])
                if self.normalization:
                    features[:, column] = MinMaxScaler()(features[:, column])
            # 离散值处理.
            else:
                pass

        self.x = features

    def _preprocessing_binary_labels(self, labels):
        """预处理二值标签, 将标签转换为数值化的稀疏向量.

        Arguments:
            labels: numpy.ndarray, 原始的标签.
        """
        _raw_labels = np.unique(labels)

        _positive_key_words = ('是', 'y', 'yes', 'Yes', 'YES', 'Y')
        _negative_key_words = ('否', 'n', 'no', 'No', 'NO', 'N')

        if type(labels[0]) is not int:
            for key_word in _positive_key_words:
                labels[labels == key_word] = 1
            for key_word in _negative_key_words:
                labels[labels == key_word] = 0

        self.y = labels.astype(int)
        self.class_indices = {_raw_labels[0]: 0, _raw_labels[1]: 1}

    def _preprocessing_categorical_babels(self, labels):
        """预处理多分类标签, 将标签转换为数值化的稀疏向量.

            Arguments:
                labels: numpy.ndarray, 原始的标签.
        """
        _raw_labels = np.unique(labels)

        if type(labels[0]) is not int:
            for index, raw_label in enumerate(_raw_labels):
                labels[labels == raw_label] = index
                self.class_indices.update({raw_label: index})

        self.y = labels.astype(int)
