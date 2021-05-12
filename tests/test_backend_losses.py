"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍不具有实际意义.

    Notes:
        - 在Python中print(0.1+0.2)输出不为0.3, activations后端引擎涉及到大量的计算梯度,
        导致这种差异尤为明显, 但实际调试发现差异小于1e-15, 对实际结果影响在工程上可以忽略,
        因此设置阈值为1e-14.
"""
import numpy as np

from classicML.backend.cc.losses import Loss as CcLoss
from classicML.backend.cc.losses import Crossentropy as CcCrossentropy
from classicML.backend.cc.losses import LogLikelihood as CcLogLikelihood
from classicML.backend.cc.losses import MeanSquaredError as CcMeanSquaredError
from classicML.backend.cc.losses import MSE as CcMSE

from classicML.backend.python.losses import Loss as PyLoss
from classicML.backend.python.losses import Crossentropy as PyCrossentropy
from classicML.backend.python.losses import LogLikelihood as PyLogLikelihood
from classicML.backend.python.losses import MeanSquaredError as PyMeanSquaredError
from classicML.backend.python.losses import MSE as PyMSE

THRESHOLD = 1e-14


class TestLoss(object):
    def test_loss(self):
        cc_loss = CcLoss()
        py_loss = PyLoss()

        assert cc_loss.name == py_loss.name

    def test_crossentropy(self):
        cc_ce = CcCrossentropy('cc_ce')
        py_ce = PyCrossentropy('py_ce')

        binary_y_pred = np.random.random(size=[10, 1])
        binary_y_true = np.random.random(size=[10, 1])

        cat_y_pred = np.random.random(size=[10, 5])
        cat_y_true = np.random.randint(low=0, high=5, size=10)
        cat_y_true = np.eye(5)[cat_y_true]

        assert cc_ce.name != py_ce.name
        assert abs(cc_ce(binary_y_pred, binary_y_true) - py_ce(binary_y_pred, binary_y_true)) <= THRESHOLD
        assert abs(cc_ce(cat_y_pred, cat_y_true) - py_ce(cat_y_pred, cat_y_true)) <= THRESHOLD

    def test_log_likelihood(self):
        cc_ll = CcLogLikelihood('cc_ll')
        py_ll = PyLogLikelihood('py_ll')

        y_true = np.random.randint(low=0, high=2, size=[17, 1])
        beta = np.random.random(size=[3, 1])
        x_hat = np.random.random(size=[17, 3])

        assert cc_ll.name != py_ll.name
        assert abs(cc_ll(y_true, beta, x_hat) - py_ll(y_true, beta, x_hat)) <= THRESHOLD

    def test_mean_squared_error_mse(self):
        cc_mean_squared_error = CcMeanSquaredError('cc_mean_squared_error')
        py_mean_squared_error = PyMeanSquaredError('py_mean_squared_error')
        cc_mse = CcMSE('cc_mse')
        py_mse = PyMSE('py_mse')

        y_pred = np.random.random(size=[10, 1])
        y_true = np.random.randint(low=0, high=2, size=[10, 1])

        assert cc_mean_squared_error.name != py_mean_squared_error.name
        assert cc_mse.name != py_mse.name
        assert abs(cc_mean_squared_error(y_pred, y_true) - py_mean_squared_error(y_pred, y_true)) <= THRESHOLD
        assert abs(cc_mse(y_pred, y_true) - py_mse(y_pred, y_true)) <= THRESHOLD
        assert cc_mse(y_pred, y_true) == cc_mean_squared_error(y_pred, y_true)
