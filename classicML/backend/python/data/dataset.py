"""classicML中的用来组织数据集模块."""
import numpy as np
import pandas as pd

from classicML import CLASSICML_LOGGER
from classicML.backend.python.data.preprocessing import DummyEncoder
from classicML.backend.python.data.preprocessing import Imputer
from classicML.backend.python.data.preprocessing import MaxMarginEncoder
from classicML.backend.python.data.preprocessing import MinMaxScaler
from classicML.backend.python.data.preprocessing import OneHotEncoder
from classicML.backend.python.data.preprocessing import StandardScaler


class Dataset(object):
    """数据集,
    数据集提供了对输入数据的预处理和封装的功能, 使之满足cml模型输入的需要.

    Attributes:
        dataset_type: {'train', 'validation', 'test'}, default='train',
            数据集的类型, 如果声明为测试集, 将不会生成对应的标签.
        label_mode: {'one-hot', 'max-margin'}, default=None,
            标签的编码格式.
        fillna: bool, default=True,
            是否填充缺失值.
        digitization: bool, default=False,
            是否使用数值化, 将离散标签转化成数值.
        normalization: bool, default=False,
            是否使用归一化.
        standardization: bool, default=False,
            是否使用标准化.
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
                 digitization=False,
                 normalization=False,
                 standardization=False,
                 name=None):
        """初始化数据集.

        Arguments:
            dataset_type: {'train', 'validation', 'test'}, default='train',
                数据集的类型, 如果声明为测试集, 将不会生成对应的标签.
            label_mode: {'one-hot', 'max-margin'}, default=None,
                标签的编码格式.
            fillna: bool, default=True,
                是否填充缺失值.
            digitization: bool, default=False,
                是否使用数值化, 将离散标签转化成数值.
            normalization: bool, default=False,
                是否使用归一化.
            standardization: bool, default=False,
                是否使用标准化.
            name: str, default=None,
                数据集的名称.
        """
        super(Dataset, self).__init__()
        self.dataset_type = dataset_type.lower()
        self.label_mode = label_mode
        self.fillna = fillna
        self.digitization = digitization
        self.normalization = normalization
        self.standardization = standardization
        self.name = name

        self.x = None
        self.y = None
        self.class_indices = dict()

    def from_dataframe(self, dataframe):
        """从DataFrame中加载数据集.

        Arguments:
            dataframe: pandas.DataFrame, 原始的数据.

        Returns:
            经过预处理的特征数据和标签.
        """
        # 预处理特征数据.
        self._preprocessing_features(dataframe.iloc[:, :-1].values)
        # 预处理标签.
        if self.dataset_type != 'test':
            if len(np.unique(dataframe.iloc[:, -1].values)) == 2:
                self._preprocessing_binary_labels(dataframe.iloc[:, -1].values)
            else:
                self._preprocessing_categorical_labels(dataframe.iloc[:, -1].values)
            # 编码标签.
            self._encoder_label()

        return self.x, self.y

    def from_csv(self, filepath, sep=','):
        """从CSV文件中加载数据集,
         也可以从其他的结构化文本读入数据, 例如TSV等.

        Arguments:
            filepath: str, CSV文件的路径.
            sep: str, default=',',
                使用的文本分隔符.

        Returns:
            经过预处理的特征数据和标签.
        """
        data = pd.read_csv(filepath_or_buffer=filepath,
                           sep=sep,
                           index_col=0,
                           header=0)
        self.from_dataframe(data)

        return self.x, self.y

    def from_tensor_slices(self, x, y=None):
        """从张量流加载数据集.

        Arguments:
            x: array-like,
                处理后的特征数据.
            y: array-like, default=None,
                处理后的标签.

        Returns:
            经过预处理的特征数据和标签.
        """
        if y is None and self.dataset_type == 'train':
            CLASSICML_LOGGER.warn('请检查您的数据集, 训练集似乎应该有标签数据.')

        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        # 预处理特征数据.
        self._preprocessing_features(x)

        # 预处理标签.
        if self.dataset_type != 'test':
            if y.shape[1] > 2:
                self._preprocessing_categorical_labels(y)
            else:
                self._preprocessing_binary_labels(y)
            self._encoder_label()

        return self.x, self.y

    def _preprocessing_features(self, features):
        """预处理特征值.

        Arguments:
            features: numpy.ndarray, 特征数据.
        """
        # 处理缺失值.
        if self.fillna:
            features = Imputer()(features)

        _del_list = list()
        for column in range(features.shape[1]):
            try:
                # 连续值处理.
                _current_feature = np.asarray(features[:, column], dtype='float32')
                if self.standardization:
                    features[:, column] = StandardScaler(axis=0)(_current_feature)
                if self.normalization:
                    features[:, column] = MinMaxScaler()(_current_feature)
            except ValueError:
                # 离散值处理.
                if self.digitization:
                    features = np.c_[features, DummyEncoder()(features[:, column])]
                    _del_list.append(column)

        # 删除被数值化的列.
        if len(_del_list) > 0:
            features = np.delete(features, _del_list, axis=1)

        self.x = features

    def _preprocessing_binary_labels(self, labels):
        """预处理二值标签, 将标签转换为数值化的稀疏向量.

        Arguments:
            labels: numpy.ndarray, 原始的标签.
        """
        _raw_labels = np.unique(labels)

        _positive_key_words = ('是', 'y', 'yes', 'Yes', 'YES', 'Y')
        _negative_key_words = ('否', 'n', 'no', 'No', 'NO', 'N')

        if type(labels[0]) not in (int, np.ndarray):
            for key_word in _positive_key_words:
                labels[labels == key_word] = 1
            for key_word in _negative_key_words:
                labels[labels == key_word] = 0

        self.y = labels.astype(int)
        self.class_indices = {_raw_labels[0]: 0, _raw_labels[1]: 1}

    def _preprocessing_categorical_labels(self, labels):
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

    def _encoder_label(self):
        """编码标签."""
        if self.label_mode == 'one-hot':
            self.y, _class_indices = OneHotEncoder()(self.y)
            self._set_real_class_indices(self.class_indices, _class_indices)
        elif self.label_mode == 'max-margin':
            self.y, _class_indices = MaxMarginEncoder()(self.y)
            self._set_real_class_indices(self.class_indices, _class_indices)

    def _set_real_class_indices(self, preprocessed_class_indices, encoder_class_indices):
        """设置真实的类标签和类索引的映射字典.

        Arguments:
            preprocessed_class_indices: dict, 预处理后的类标签和类索引的映射字典.
            encoder_class_indices: dict, 编码器的类标签和类索引的映射字典.
        """
        class_indices = dict()
        for key, value in preprocessed_class_indices.items():
            class_indices.update({key: encoder_class_indices[value]})

        self.class_indices = class_indices
