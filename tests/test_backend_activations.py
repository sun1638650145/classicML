"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.
"""
import numpy as np

from classicML.backend.cc.activations import Activation as CcActivation
from classicML.backend.cc.activations import Relu as CcRelu

from classicML.backend.python.activations import Activation as PyActivation
from classicML.backend.python.activations import Relu as PyRelu


class TestActivation(object):
    def test_activation(self):
        cc_activation = CcActivation()
        py_activation = PyActivation()

        assert cc_activation.name == py_activation.name

    def test_relu(self):
        cc_relu = CcRelu(name='cc_relu')
        py_relu = PyRelu(name='py_relu')

        assert cc_relu.name != py_relu.name
