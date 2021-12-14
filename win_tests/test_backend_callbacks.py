import numpy as np

from classicML.backend.cc.callbacks import History as CcHistory
from classicML.backend.python.callbacks import History as PyHistory


class TestCallback(object):
    def test_history(self):
        cc_history = CcHistory()
        py_history = PyHistory()

        loss_values = np.random.random(10)
        metric_values = np.random.random(10)

        for loss_value, metric_value in zip(loss_values, metric_values):
            cc_history(loss_value, metric_value)
            py_history(loss_value, metric_value)

        assert cc_history.name == py_history.name
        assert cc_history.loss_name == py_history.loss_name
        assert cc_history.metric_name == py_history.metric_name
        assert cc_history.loss == py_history.loss
        assert cc_history.metric == py_history.metric
