from pickle import loads, dumps

import numpy as np
import pandas as pd

from classicML import CLASSICML_LOGGER
from classicML.api.models import BaseModel
from classicML.backend import get_conditional_probability
from classicML.backend import get_dependent_prior_probability
from classicML.backend import get_probability_density
from classicML.backend import type_of_target
from classicML.backend import io


class OneDependentEstimator(BaseModel):
    """独依赖估计器的基类.

    Attributes:
        attribute_name: list of name, default=None,
            属性的名称.
        is_trained: bool, default=False,
            模型训练后将被标记为True.
        is_loaded: bool, default=False,
            如果模型加载了权重将被标记为True.

    Raises:
        NotImplementedError: compile, fit, predict方法需要用户实现.
    """
    def __init__(self, attribute_name=None):
        """初始化独依赖估计器.

        Arguments:
            attribute_name: list of name, default=None,
                属性的名称.
        """
        super(OneDependentEstimator, self).__init__()
        self.attribute_name = attribute_name

        self.is_trained = False
        self.is_loaded = False

    def compile(self, *args, **kwargs):
        """编译独依赖估计器.
        """
        raise NotImplementedError

    def fit(self, x, y, **kwargs):
        """训练独依赖估计器.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            y: numpy.ndarray or pandas.DataFrame, array-like, 标签.
        """
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """使用独依赖估计器进行预测.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
        """
        raise NotImplementedError

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            KeyError: 模型权重加载失败.

        Notes:
            模型将不会加载关于优化器的超参数.
        """
        raise NotImplementedError

    def save_weights(self, filepath):
        """将模型权重保存为一个HDF5文件.

        Arguments:
            filepath: str, 权重文件保存的路径.

        Raises:
            TypeError: 模型权重保存失败.

        Notes:
            模型将不会保存关于优化器的超参数.
        """
        raise NotImplementedError


class SuperParentOneDependentEstimator(OneDependentEstimator):
    """超父独依赖估计器.

    Attributes:
        attribute_name: list of name, default=None,
            属性的名称.
        super_parent_name: str, default=None,
            超父的名称.
        super_parent_index: int, default=None,
            超父的索引值.
        _list_of_p_c: list,
            临时保存中间的概率依赖数据.
        smoothing: bool, default=None,
            是否使用平滑, 这里的实现是拉普拉斯修正.
    """
    def __init__(self, attribute_name=None):
        """初始化超父独依赖估计器.

        Arguments:
            attribute_name: list of name, default=None,
                属性的名称.
        """
        super(SuperParentOneDependentEstimator, self).__init__(attribute_name=attribute_name)

        self.super_parent_name = None
        self.super_parent_index = None
        self.smoothing = None

        self._list_of_p_c = list()

    def compile(self, super_parent_name, smoothing=True):
        """编译超父独依赖估计器.

        Arguments:
            super_parent_name: str, default=None,
                超父的名称.
            smoothing: bool, default=True,
                是否使用平滑, 这里的实现是拉普拉斯修正.
        """
        self.super_parent_name = super_parent_name
        self.smoothing = smoothing

    def fit(self, x, y, **kwargs):
        """训练超父独依赖估计器.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            y: numpy.ndarray or pandas.DataFrame, array-like, 标签.

        Returns:
            SuperParentOneDependentEstimator实例.
        """
        if isinstance(x, np.ndarray) and self.attribute_name is None:
            CLASSICML_LOGGER.warn("属性名称缺失, 请使用pandas.DataFrame; 或检查 self.attributes_name")

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)
        y = pd.Series(y)
        y.reset_index(drop=True, inplace=True)

        for index, feature_name in enumerate(x.columns):
            if self.super_parent_name == feature_name:
                self.super_parent_index = index

        for category in np.unique(y):
            unique_values_xi = x.iloc[:, self.super_parent_index].unique()
            for value in unique_values_xi:
                # 初始化概率字典.
                p_c = dict()

                # 获取有依赖的类先验概率P(c, xi).
                c_xi = (x.values[:, self.super_parent_index] == value) & (y == category)
                c_xi = x.values[c_xi, :]
                p_c_xi = get_dependent_prior_probability(len(c_xi),
                                                         len(x.values),
                                                         len(unique_values_xi),
                                                         self.smoothing)
                p_c.update({'p_c_xi': p_c_xi})

                # 获取有依赖的类条件概率P(xj|c, xi)或概率密度p(xj|c, xi)所需的信息.
                for attribute in range(x.shape[1]):
                    xj = x.iloc[:, attribute]
                    continuous = type_of_target(xj.values) == 'continuous'

                    if continuous:
                        # 连续值概率密度函数信息.
                        if len(c_xi) <= 2:
                            # 样本数量过少的时候, 使用全局的均值和方差.
                            mean = np.mean(x.values[y == category, attribute])
                            var = np.var(x.values[y == category, attribute])
                        else:
                            mean = np.mean(c_xi[:, attribute])
                            var = np.var(c_xi[:, attribute])
                        p_c.update({x.columns[attribute]: {
                                    'continuous': continuous,
                                    'values': [mean, var]}})
                    else:
                        # 离散值条件概率信息.
                        unique_value = xj.unique()
                        num_of_unique_value = len(unique_value)
                        value_count = pd.DataFrame(np.zeros((1, num_of_unique_value)), columns=unique_value)

                        for key in pd.value_counts(c_xi[:, attribute]).keys():
                            value_count[key] += pd.value_counts(c_xi[:, attribute])[key]

                        # 统计不同属性值的样本总数.
                        D_c_xi = dict()
                        for name in value_count:
                            D_c_xi.update({name: float(value_count[name].values)})

                        p_c.update({x.columns[attribute]: {
                                    'continuous': continuous,
                                    'values': [D_c_xi, c_xi.shape[0], num_of_unique_value],
                                    'smoothing': self.smoothing}})

                self._list_of_p_c.append({'category': category, 'attribute': value, 'p_c': p_c})

        self.is_trained = True

        return self

    def predict(self, x, probability=False):
        """使用超父独依赖估计器进行预测.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            probability: bool, default=False,
                是否使用归一化的概率形式.

        Returns:
            SuperParentOneDependentEstimator的预测结果,
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

        y_pred = list()

        if len(x.shape) == 1:
            p_0, p_1 = self._predict(x)
            if probability:
                y_pred.append([p_0 / (p_0 + p_1), p_1 / (p_0 + p_1)])
            else:
                if p_0 > p_1:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
        else:
            for i in range(x.shape[0]):
                x_test = x.iloc[i, :]
                p_0, p_1 = self._predict(x_test)
                if probability:
                    y_pred.append([p_0 / (p_0 + p_1), p_1 / (p_0 + p_1)])
                else:
                    if p_0 > p_1:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)

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
                                                   model_name='SuperParentOneDependentEstimator')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.super_parent_name = compile_ds.attrs['super_parent_name']
            self.smoothing = compile_ds.attrs['smoothing']
            self._list_of_p_c = loads(weights_ds.attrs['_list_of_p_c'].tobytes())

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
                                                   model_name='SuperParentOneDependentEstimator')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['super_parent_name'] = self.super_parent_name
            compile_ds.attrs['smoothing'] = self.smoothing
            weights_ds.attrs['_list_of_p_c'] = np.void(dumps(self._list_of_p_c))
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')

    def _predict(self, x, attribute_list=None, super_parent_index=None):
        """通过平均独依赖估计器预测单个样本.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like,
                特征数据.
            attribute_list: list, default=None,
                临时保存中间的概率依赖数据(包含所有的属性, 仅使用AODE时有意义).
            super_parent_index: int, default=None,
                超父的索引值.

        Returns:
            返回预测的结果.
        """
        y_pred = [0.0, 0.0]

        if attribute_list is None and super_parent_index is None:
            for i in self._list_of_p_c:
                self._calculate_posterior_probability(x, i, y_pred)
        else:
            # TODO(Steve R. Sun, tag:performance): 这里是为了满足AODE调用的便利.
            for i in attribute_list[super_parent_index]:
                if i['attribute'] == x[super_parent_index]:
                    self._calculate_posterior_probability(x, i, y_pred)

        return y_pred

    @staticmethod
    def _calculate_posterior_probability(x, i, y_pred):
        """计算后验概率.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like,
                特征数据.
            i: int, 样本的索引.
            y_pred: list, 后验概率列表.
        """
        _p_c = i['p_c']

        if i['category'] == 0:
            for index, probability in enumerate(_p_c):
                if probability == 'p_c_xi':
                    # 先添加P(c, xi)
                    y_pred[0] += np.log(_p_c[probability])
                else:
                    # 添加P(xj|c, xi)
                    continuous = _p_c[probability]['continuous']
                    # 分别处理连续值和离散值.
                    if continuous:
                        mean, var = _p_c[probability]['values']
                        probability_density = get_probability_density(x[index - 1], mean, var)
                        y_pred[0] += np.log(probability_density)  # 存放数据中多存放一个p_c_xi导致和x的索引无法对齐.
                    else:
                        D_c_xi_xj, D_c_xi, num_of_unique_value = _p_c[probability]['values']
                        y_pred[0] += np.log(get_conditional_probability(D_c_xi_xj[x[index - 1]],
                                                                        D_c_xi,
                                                                        num_of_unique_value,
                                                                        _p_c[probability]['smoothing']))
        elif i['category'] == 1:
            for index, probability in enumerate(_p_c):
                if probability == 'p_c_xi':
                    y_pred[1] += np.log(_p_c[probability])
                else:
                    continuous = _p_c[probability]['continuous']
                    if continuous:
                        mean, var = _p_c[probability]['values']
                        probability_density = get_probability_density(x[index - 1], mean, var)
                        y_pred[1] += np.log(probability_density)
                    else:
                        D_c_xi_xj, D_c_xi, num_of_unique_value = _p_c[probability]['values']
                        y_pred[1] += np.log(get_conditional_probability(D_c_xi_xj[x[index - 1]],
                                                                        D_c_xi,
                                                                        num_of_unique_value,
                                                                        _p_c[probability]['smoothing']))


class AveragedOneDependentEstimator(SuperParentOneDependentEstimator):
    """平均独依赖估计器.

    Attributes:
        attribute_name: list of name, default=None,
            属性的名称.
        super_parent_name: str, default=None,
            超父的名称.
        smoothing: bool, default=None,
            是否使用平滑, 这里的实现是拉普拉斯修正.
        m: int, default=0,
            阈值常数, 样本小于此值的属性将不会被作为超父类.
        _attribute_list: list,
            临时保存中间的概率依赖数据(包含所有的属性).
    """
    def __init__(self, attribute_name=None):
        """初始化平均独依赖估计器.

        Arguments:
            attribute_name: list of name, default=None,
                属性的名称.
        """
        super(AveragedOneDependentEstimator, self).__init__(attribute_name=attribute_name)

        self.smoothing = None
        self.m = 0
        self._attribute_list = list()

    def compile(self, smoothing=True, m=0, **kwargs):
        """编译平均独依赖估计器.

        Arguments:
            smoothing: bool, default=True,
                是否使用平滑, 这里的实现是拉普拉斯修正.
            m: int, default=0,
                阈值常数, 样本小于此值的属性将不会被作为超父类.
        """
        self.smoothing = smoothing
        self.m = m

    def fit(self, x, y, **kwargs):
        """训练平均独依赖估计器.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.
            y: numpy.ndarray or pandas.DataFrame, array-like, 标签.

        Returns:
            AverageOneDependentEstimator实例.
        """
        if isinstance(x, np.ndarray) and self.attribute_name is None:
            CLASSICML_LOGGER.warn("属性名称缺失, 请使用pandas.DataFrame; 或检查 self.attributes_name")

        # TODO(Steve R. Sun, tag:code): 暂时没有找到合理的断点续训的理论支持.
        self._attribute_list = list()

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)
        y = pd.Series(y)
        y.reset_index(drop=True, inplace=True)

        number_of_samples, number_of_attributes = x.shape

        # 获取离散属性的全部取值.
        discrete_unique_values = dict()
        for attribute in range(number_of_attributes):
            xi = x.iloc[:, attribute]
            if (type_of_target(xi.values) != 'continuous') and (pd.value_counts(xi).values > self.m).all():
                discrete_unique_values.update({x.columns[attribute]: xi.unique()})

        # 每个属性作为超父类构建SPODE.
        for index, key in enumerate(discrete_unique_values.keys()):
            self.super_parent_name = key
            super(AveragedOneDependentEstimator, self).fit(x, y)
            current_attribute_list = self._list_of_p_c

            self._attribute_list.append(current_attribute_list)

        self.is_trained = True

        return self

    def predict(self, x, **kwargs):
        """使用平均独依赖估计器进行预测.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.

        Returns:
            AverageOneDependentEstimator的预测结果.

        Raises:
            ValueError: 模型没有训练的错误.
        """
        if self.is_trained is False and self.is_loaded is False:
            CLASSICML_LOGGER.error('模型没有训练')
            raise ValueError('你必须先进行训练')

        # 为特征数据添加属性信息.
        x = pd.DataFrame(x, columns=self.attribute_name)
        x.reset_index(drop=True, inplace=True)

        y_pred = list()
        if len(x.shape) == 1:
            y_pred.append(self._predict(x))
        else:
            for i in range(x.shape[0]):
                x_test = x.iloc[i, :]
                y_pred.append(self._predict(x_test))

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
                                                   model_name='AveragedOneDependentEstimator')
        # 加载模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            self.smoothing = compile_ds.attrs['smoothing']
            self.m = compile_ds.attrs['m']
            self._attribute_list = loads(weights_ds.attrs['_attribute_list'].tobytes())

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
                                                   model_name='AveragedOneDependentEstimator')
        # 保存模型参数.
        try:
            compile_ds = parameters_gp['compile']
            weights_ds = parameters_gp['weights']

            compile_ds.attrs['smoothing'] = self.smoothing
            compile_ds.attrs['m'] = self.m
            weights_ds.attrs['_attribute_list'] = np.void(dumps(self._attribute_list))
        except TypeError:
            CLASSICML_LOGGER.error('模型权重保存失败, 请检查文件是否损坏')
            raise TypeError('模型权重保存失败')

    def _predict(self, x, **kwargs):
        """
        通过平均独依赖估计器预测单个样本.

        Arguments:
            x: numpy.ndarray or pandas.DataFrame, array-like, 特征数据.

        Returns:
            返回预测的结果.
        """
        avg_result = {'0': 0.0, '1': 0.0}
        for i in range(len(self._attribute_list)):
            _temp_y_pred = super(AveragedOneDependentEstimator, self)._predict(x, self._attribute_list, i)
            avg_result['0'] += _temp_y_pred[0]
            avg_result['1'] += _temp_y_pred[1]

        if avg_result['0'] > avg_result['1']:
            return 0
        else:
            return 1
