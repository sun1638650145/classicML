import numpy as np


def beta_initializer(attr_of_x):
    """"""
    beta = np.random.randn(attr_of_x + 1, 1) * 0.5 + 1

    return beta
