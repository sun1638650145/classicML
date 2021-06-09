from pickle import loads, dumps

import numpy as np
import pandas as pd

from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import type_of_target
from classicML.backend import get_conditional_probability
from classicML.backend import get_prior_probability
from classicML.backend import get_probability_density
from classicML.backend import io


class NaiveBayesClassifier(BaseModel):
    """朴素贝叶斯分类器.

    Attributes:
        attribute_name: list of name, default=None,
            属性的名称.
        smoothing: bool, default=True,
            是否使用平滑.
        p_0: float, 反例的类先验概率.
        p_1: float, 正例的类先验概率.
        pi_0: dict, 反例的类条件概率(概率密度).
        pi_1: dict, 正例的类条件概率(概率密度).
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.
    """
    def __init__(self, attribute_name=None):
        """初始化朴素贝叶斯分类器.

        Arguments:
            attribute_name: list of name, default=None,
                属性的名称.
        """
        super(NaiveBayesClassifier, self).__init__()
        self.attribute_name = attribute_name

        self.p_0 = None
        self.p_1 = None
        self.pi_0 = dict()
        self.pi_1 = dict()
        self.smoothing = None
        self.is_trained = False
        self.is_loaded = False

    def compile(self, smoothing=True):
        """编译朴素贝叶斯分类器.

        Arguments:
            smoothing: bool, default=True,
                是否使用平滑, 这里的实现是拉普拉斯修正.
        """
        self.smoothing = smoothing

    def fit(self, x, y, **kwargs):
        """训练朴素贝叶斯分类器.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据
            y: numpy.ndarray or pandas.DataFrame, array-like, 标签.

        Returns:
            NaiveBayesClassifier实例.
        """
        if isinstance(x, np.ndarray) and self.attribute_name is None:
            CLASSICML_LOGGER.warn("属性名称缺失, 请使用pandas.DataFrame; 或检查 self.attributes_name")

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)
        y = pd.Series(y)
        y.reset_index(drop=True, inplace=True)

        # 获取反正例的样本总数.
        negative_samples = x[y == 0]
        positive_samples = x[y == 1]
        num_of_negative_samples = len(negative_samples)
        num_of_positive_samples = len(positive_samples)

        # 获取类先验概率P(c).
        self.p_0, self.p_1 = get_prior_probability(len(x.values), y.values, self.smoothing)

        number_of_samples, number_of_attributes = x.shape
        # 获取每个属性类条件概率P(x_i|c)或类概率密度p(x_i|c)所需的信息.
        for attribute in range(number_of_attributes):
            xi = x.iloc[:, attribute]
            continuous = (type_of_target(xi.values) == 'continuous')

            xi0 = negative_samples.iloc[:, attribute]
            xi1 = positive_samples.iloc[:, attribute]
            if continuous:
                # 连续值概率密度函数信息.
                xi0_mean = np.mean(xi0)
                xi1_mean = np.mean(xi1)

                xi0_var = np.var(xi0)
                xi1_var = np.var(xi1)

                self.pi_0.update({x.columns[attribute]: {
                                  'continuous': continuous,
                                  'values': [xi0_mean, xi0_var]}})  # values存放了均值和方差.
                self.pi_1.update({x.columns[attribute]: {
                                  'continuous': continuous,
                                  'values': [xi1_mean, xi1_var]}})
            else:
                # 离散值计算条件概率信息.
                unique_value = xi.unique()
                num_of_unique_value = len(unique_value)

                xi0_value_count = pd.DataFrame(np.zeros((1, num_of_unique_value)), columns=unique_value)
                xi1_value_count = pd.DataFrame(np.zeros((1, num_of_unique_value)), columns=unique_value)

                for key in pd.value_counts(xi0).keys():
                    xi0_value_count[key] += pd.value_counts(xi0)[key]
                for key in pd.value_counts(xi1).keys():
                    xi1_value_count[key] += pd.value_counts(xi1)[key]

                # 统计不同属性值的样本总数.
                D_c_xi0 = dict()
                D_c_xi1 = dict()
                for index, name in enumerate(pd.value_counts(xi).keys()):
                    D_c_xi0.update({name: np.squeeze(xi0_value_count.values)[index]})
                    D_c_xi1.update({name: np.squeeze(xi1_value_count.values)[index]})

                self.pi_0.update({x.columns[attribute]: {
                    'continuous': continuous,
                    # values存放了每个样本的数量, 在某个类别上的样本总数, 类别的数量.
                    'values': [D_c_xi0, num_of_negative_samples, num_of_unique_value],
                    'smoothing': self.smoothing}})
                self.pi_1.update({x.columns[attribute]: {
                    'continuous': continuous,
                    'values': [D_c_xi1, num_of_positive_samples, num_of_unique_value],
                    'smoothing': self.smoothing}})

        self.is_trained = True

        return self

    def predict(self, x, probability=False, **kwargs):
        """使用朴素贝叶斯分类器进行预测.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            probability: bool, default=False,
                是否使用归一化的概率形式.

        Returns:
            NaiveBayesClassifier的预测结果,
            不使用概率形式将返回0或1的标签数组, 使用将返回反正例概率的数组.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)

        # 避免下溢进行对数处理.
        p_0 = np.log(self.p_0)
        p_1 = np.log(self.p_1)

        y_pred = list()
        if len(x.shape) == 1:
            y_pred.append(self._predict(x, p_0, p_1, probability))
        else:
            for i in range(x.shape[0]):
                x_test = x.iloc[i, :]
                y_pred.append(self._predict(x_test, p_0, p_1, probability))

        return y_pred

    def _predict(self, x, p_0, p_1, probability):
        """通过朴素贝叶斯分类器预测单个样本.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            p_0: float, 反例的类先验概率.
            p_1: float, 正例的类先验概率.
            probability: bool, default=False,
                是否使用归一化的概率形式.

        Returns:
            返回预测的结果.
        """
        for index, attribute in enumerate(self.attribute_name):
            pxi_0 = self.pi_0[attribute]
            pxi_1 = self.pi_1[attribute]

            # 正例连续反例必然连续.
            if pxi_0['continuous']:
                mean0, var0 = pxi_0['values']
                mean1, var1 = pxi_1['values']

                p_0 += np.log(get_probability_density(x[index], mean0, var0))
                p_1 += np.log(get_probability_density(x[index], mean1, var1))
            else:
                D_c_x0, D_c0, N0 = pxi_0['values']
                D_c_x1, D_c1, N1 = pxi_1['values']

                p_0 += np.log(get_conditional_probability(D_c_x0[x[index]], D_c0, N0, self.smoothing))
                p_1 += np.log(get_conditional_probability(D_c_x1[x[index]], D_c1, N1, self.smoothing))

        if probability:
            return [p_0 / (p_0 + p_1), p_1 / (p_0 + p_1)]
        else:
            if p_0 > p_1:
                return 0
            else:
                return 1

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
                                                   model_name='NaiveBayesClassifier')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.smoothing = compile_ds.attrs['smoothing']
            self.p_0 = weights_ds.attrs['p_0']
            self.p_1 = weights_ds.attrs['p_1']
            self.pi_0 = loads(weights_ds.attrs['pi_0'].tobytes())
            self.pi_1 = loads(weights_ds.attrs['pi_1'].tobytes())

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
                                                   model_name='NaiveBayesClassifier')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['smoothing'] = self.smoothing
            weights_ds.attrs['p_0'] = self.p_0
            weights_ds.attrs['p_1'] = self.p_1
            weights_ds.attrs['pi_0'] = np.void(dumps(self.pi_0))
            weights_ds.attrs['pi_1'] = np.void(dumps(self.pi_1))
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')
