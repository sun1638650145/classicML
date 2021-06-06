"""classicML的初始化器."""
import numpy as np


class Initializer(object):
    """初始化器的基类.

    Attributes:
        name: str, default='initializer',
            初始化器的名称.
        seed: int, default=None,
            初始化器的随机种子.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
    """
    def __init__(self, name='initializer', seed=None):
        """
        Arguments:
            name: str, default='initializer',
                初始化器的名称.
            seed: int, default=None,
                初始化器的随机种子.
        """
        self.name = name
        self.seed = seed
        np.random.seed(self.seed)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RandomNormal(Initializer):
    """正态分布随机初始化器.
    """
    def __init__(self, name='random_normal', seed=None):
        super(RandomNormal, self).__init__(name=name, seed=seed)

    def __call__(self, attributes_or_structure):
        """函数实现.

        Arguments:
            attributes_or_structure: int or list,
                如果是逻辑回归就是样本的特征数;
                如果是神经网络, 就是定义神经网络的网络结构.
        """
        if isinstance(attributes_or_structure, int):
            parameters = np.random.randn(attributes_or_structure + 1, 1)  # 初始化属性数+1(偏置项b)
        else:
            parameters = {}
            num_of_layers = len(attributes_or_structure)

            for layer in range(num_of_layers - 1):
                w = np.random.randn(attributes_or_structure[layer + 1], attributes_or_structure[layer])
                b = np.zeros((1, attributes_or_structure[layer + 1]))
                parameters['w' + str(layer + 1)] = w
                parameters['b' + str(layer + 1)] = b

        return parameters


class HeNormal(Initializer):
    """He正态分布随机初始化器.

    References:
        - [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
          ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
    """
    def __init__(self, name='he_normal', seed=None):
        super(HeNormal, self).__init__(name=name, seed=seed)

    def __call__(self, attributes_or_structure):
        """初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

        Arguments:
            attributes_or_structure: int or list,
                如果是逻辑回归就是样本的特征数;
                如果是神经网络, 就是定义神经网络的网络结构.
        """
        if isinstance(attributes_or_structure, int):
            parameters = (np.random.randn(attributes_or_structure + 1, 1)
                          * np.sqrt(2 / attributes_or_structure))  # 初始化属性数+1(偏置项b)
        else:
            parameters = {}
            num_of_layers = len(attributes_or_structure)

            for layer in range(num_of_layers - 1):
                w = (np.random.randn(attributes_or_structure[layer + 1], attributes_or_structure[layer])
                     * np.sqrt(2 / attributes_or_structure[layer]))
                b = np.zeros((1, attributes_or_structure[layer + 1]))
                parameters['w' + str(layer + 1)] = w
                parameters['b' + str(layer + 1)] = b

        return parameters


class XavierNormal(Initializer):
    """Xavier正态分布随机初始化器,
        也叫做Glorot正态分布随机初始化器.

    References:
        - [Glorot et al., 2010](https://proceedings.mlr.press/v9/glorot10a.html)
          ([pdf](https://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
    """
    def __init__(self, name='xavier_normal', seed=None):
        super(XavierNormal, self).__init__(name=name, seed=seed)

    def __call__(self, attributes_or_structure):
        """初始化方式为W~N(0, sqrt(2/N_in+N_out)),
            其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

        Arguments:
            attributes_or_structure: int or list,
                如果是逻辑回归就是样本的特征数;
                如果是神经网络, 就是定义神经网络的网络结构.
        """
        if isinstance(attributes_or_structure, int):
            # 逻辑回归没有多层结构
            # 令 2 / (N_in + N_out) = 2 / N_in * 2, 即 N_in
            parameters = (np.random.randn(attributes_or_structure + 1, 1)
                          * np.sqrt(attributes_or_structure))
        else:
            parameters = {}
            num_of_layers = len(attributes_or_structure)

            for layer in range(num_of_layers - 1):
                w = (np.random.randn(attributes_or_structure[layer + 1], attributes_or_structure[layer])
                     * np.sqrt(2 / (attributes_or_structure[layer] + attributes_or_structure[layer + 1])))
                b = np.zeros((1, attributes_or_structure[layer + 1]))
                parameters['w' + str(layer + 1)] = w
                parameters['b' + str(layer + 1)] = b

        return parameters


class GlorotNormal(XavierNormal):
    """Glorot正态分布随机初始化器.
        具体实现参看XavierNormal.
    """
    def __init__(self, name='glorot_normal', seed=None):
        super(GlorotNormal, self).__init__(name=name, seed=seed)


class RBFNormal(Initializer):
    """RBF网络的初始化器.
    """
    def __init__(self, name='rbf_normal', seed=None):
        super(RBFNormal, self).__init__(name=name, seed=seed)

    def __call__(self, hidden_units):
        """
        Arguments:
            hidden_units: int, 径向基函数网络的隐含层神经元数量.

        Notes:
            - 这里隐含层神经元中心本应用np.random.randn全部初始化,
              但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
              因此, 全部初始化为正数.
        """
        parameters = {'w': np.zeros([1, hidden_units]),
                      'b': np.zeros([1, 1]),
                      'c': np.random.rand(hidden_units, 2),  # 隐含层神经元的中心
                      'beta': np.random.randn(1, hidden_units)}  # 高斯径向基函数的系数

        return parameters
