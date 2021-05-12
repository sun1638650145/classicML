"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍不具有实际意义.

    Notes:
        - 在Python中print(0.1+0.2)输出不为0.3, activations后端引擎涉及到大量的计算梯度,
        导致这种差异尤为明显, 但实际调试发现差异小于1e-15, 对实际结果影响在工程上可以忽略,
        因此设置阈值为1e-15.
"""
import numpy as np

from classicML.backend.cc.kernels import Kernel as CcKernel
from classicML.backend.cc.kernels import Gaussian as CcGaussian
from classicML.backend.cc.kernels import Linear as CcLinear
from classicML.backend.cc.kernels import Polynomial as CcPolynomial
from classicML.backend.cc.kernels import RBF as CcRBF
from classicML.backend.cc.kernels import Sigmoid as CcSigmoid

from classicML.backend.python.kernels import Kernel as PyKernel
from classicML.backend.python.kernels import Gaussian as PyGaussian
from classicML.backend.python.kernels import Linear as PyLinear
from classicML.backend.python.kernels import Polynomial as PyPolynomial
from classicML.backend.python.kernels import RBF as PyRBF
from classicML.backend.python.kernels import Sigmoid as PySigmoid

THRESHOLD = 1e-15


class TestKernel(object):
    def test_kernel(self):
        cc_kernel = CcKernel()
        py_kernel = PyKernel()

        assert cc_kernel.name == py_kernel.name

    def test_gaussian_rbf(self):
        cc_gaussian = CcGaussian('cc_gaussian', gamma=0.5)
        py_gaussian = PyGaussian('py_gaussian', gamma=0.5)
        cc_rbf = CcRBF('cc_rbf', gamma=0.5)
        py_rbf = PyRBF('py_rbf', gamma=0.5)

        x_i = np.random.random(size=[10, 2])
        x_j = np.random.random(size=[1, 2])

        assert cc_gaussian.name != py_gaussian.name
        assert cc_rbf.name != py_rbf.name
        assert (abs(cc_gaussian(x_i, x_j) - py_gaussian(x_i, x_j)) <= THRESHOLD).all()
        assert (abs(cc_gaussian(x_i, x_j) - py_gaussian(x_i, x_j)) <= THRESHOLD).all()
        assert (cc_gaussian(x_i, x_j) == cc_rbf(x_i, x_j)).all()

    def test_linear(self):
        cc_linear = CcLinear('cc_linear')
        py_linear = PyLinear('py_linear')

        x_i = np.random.random(size=[10, 2])
        x_j = np.random.random(size=[1, 2])

        assert cc_linear.name != py_linear.name
        assert (abs(cc_linear(x_i, x_j) - py_linear(x_i, x_j)) <= THRESHOLD).all()

    def test_poly(self):
        cc_poly = CcPolynomial('cc_poly', gamma=2.0, degree=2)
        py_poly = PyPolynomial('py_poly', gamma=2.0, degree=2)

        x_i = np.random.random(size=[10, 2])
        x_j = np.random.random(size=[1, 2])

        assert cc_poly.name != py_poly.name
        assert (abs(cc_poly(x_i, x_j) - py_poly(x_i, x_j)) <= THRESHOLD).all()

    def test_sigmoid(self):
        cc_sigmoid = CcSigmoid('cc_sigmoid', gamma=2.0, beta=1.5, theta=-1.1)
        py_sigmoid = PySigmoid('py_sigmoid', gamma=2.0, beta=1.5, theta=-1.1)

        x_i = np.random.random(size=[10, 2])
        x_j = np.random.random(size=[1, 2])

        assert cc_sigmoid.name != py_sigmoid.name
        assert (abs(cc_sigmoid(x_i, x_j) - py_sigmoid(x_i, x_j)) <= THRESHOLD).all()
