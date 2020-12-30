"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.
"""
import numpy as np

from classicML.backend.cc.metrics import Metric as CcMetric
from classicML.backend.cc.metrics import Accuracy as CcAccuracy

from classicML.backend.python.metrics import Metric as PyMetric
from classicML.backend.python.metrics import Accuracy as PyAccuracy


class TestMetric(object):
    def test_metric(self):
        cc_metric = CcMetric()
        py_metric = PyMetric()

        assert cc_metric.name == py_metric.name

    def test_accuracy(self):
        cc_acc = CcAccuracy('cc_acc')
        py_acc = PyAccuracy('py_acc')

        # 数据为随机产生, 不具有任何实际意义.(Accuracy的实现就是分别调用的二分类和多分类的Accuracy, 故不单独测试.)
        binary_y_true = np.random.randint(low=0, high=2, size=[10, 1])
        binary_y_pred = np.random.random(size=[10, 1])

        cat_y_true = np.random.randint(low=0, high=5, size=10)
        cat_y_true = np.eye(5)[cat_y_true]
        cat_y_pred = np.random.randint(low=0, high=5, size=10)
        cat_y_pred = np.eye(5)[cat_y_pred]

        assert cc_acc.name == py_acc.name
        assert cc_acc(binary_y_pred, binary_y_true) == py_acc(binary_y_pred, binary_y_true)
        assert cc_acc(cat_y_pred, cat_y_true) == py_acc(cat_y_pred, cat_y_true)
