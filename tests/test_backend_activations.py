"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.

    Notes:
        - 在Python中print(0.1+0.2)输出不为0.3, activations后端引擎涉及到大量的计算梯度,
        导致这种差异尤为明显, 但实际调试发现差异小于1e-15, 对实际结果影响在工程上可以忽略,
        因此设置阈值为1e-15.
"""
import numpy as np

from classicML.backend.cc.activations import relu
from classicML.backend.cc.activations import sigmoid
from classicML.backend.cc.activations import softmax

from classicML.backend.cc.activations import Activation as CcActivation
from classicML.backend.cc.activations import Relu as CcRelu
from classicML.backend.cc.activations import Sigmoid as CcSigmoid
from classicML.backend.cc.activations import Softmax as CcSoftmax

from classicML.backend.python.activations import Activation as PyActivation
from classicML.backend.python.activations import Relu as PyRelu
from classicML.backend.python.activations import Sigmoid as PySigmoid
from classicML.backend.python.activations import Softmax as PySoftmax

THRESHOLD = 1e-15


class TestActivation(object):
    def test_activation(self):
        cc_activation = CcActivation()
        py_activation = PyActivation()

        assert cc_activation.name == py_activation.name

    def test_relu(self):
        z = np.random.random(size=[3, 2])
        output = np.random.random(size=[3, 2])
        a = np.random.random(size=[3, 2])

        cc_relu = CcRelu(name='cc_relu')
        py_relu = PyRelu(name='py_relu')

        # 激活
        cc_value = cc_relu(z)
        py_value = py_relu(z)
        # 求微分
        cc_diff = cc_relu.diff(output, a)
        py_diff = py_relu.diff(output, a)

        assert cc_relu.name != py_relu.name
        assert (abs(cc_value - py_value) <= THRESHOLD).all()
        assert (abs(cc_diff - py_diff) <= THRESHOLD).all()

    def test_sigmoid(self):
        z = np.random.random(size=[3, 2])
        output = np.random.random(size=[3, 2])
        a = np.random.random(size=[3, 2])
        y_true = np.random.random(size=[3, 2])

        cc_sigmoid = CcSigmoid(name='cc_sigmoid')
        py_sigmoid = PySigmoid(name='py_sigmoid')

        # 激活
        cc_value = cc_sigmoid(z)
        py_value = py_sigmoid(z)
        # 求微分
        cc_diff = cc_sigmoid.diff(output, a, y_true)
        py_diff = py_sigmoid.diff(output, a, y_true)

        assert cc_sigmoid.name != py_sigmoid.name
        assert (abs(cc_value - py_value) <= THRESHOLD).all()
        assert (abs(cc_diff - py_diff) <= THRESHOLD).all()

    def test_softmax(self):
        z = np.random.random(size=[3, 2])
        output = np.random.random(size=[3, 2])
        a = np.random.random(size=[3, 2])

        cc_softmax = CcSoftmax(name='cc_softmax')
        py_softmax = PySoftmax(name='py_softmax')

        # 激活
        cc_value = cc_softmax(z)
        py_value = py_softmax(z)
        # 求微分
        cc_diff = cc_softmax.diff(output, a)
        py_diff = py_softmax.diff(output, a)

        assert cc_softmax.name != py_softmax.name
        assert (abs(cc_value - py_value) <= THRESHOLD).all()
        assert (abs(cc_diff - py_diff) <= THRESHOLD).all()

    def test_instance(self):
        """测试实例化."""
        assert isinstance(relu, CcRelu)
        assert isinstance(sigmoid, CcSigmoid)
        assert isinstance(softmax, CcSoftmax)