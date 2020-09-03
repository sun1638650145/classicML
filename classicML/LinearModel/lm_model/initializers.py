import numpy as np


def beta_initializer(attr_of_x):
    """逻辑回归网络参数初始化
    Parameters
    ----------
    attr_of_x : int
        特征数据属性数
    """
    beta = np.random.randn(attr_of_x + 1, 1)  # 初始化属性数+1(偏置项b)

    return beta
