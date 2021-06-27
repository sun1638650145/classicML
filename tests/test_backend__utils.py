"""
    测试cc后端和默认python后端的运行结果是否一致.
"""
import classicML as cml

from classicML.backend.cc._utils import ProgressBar as CcProgressBar
from classicML.backend.python._utils import ProgressBar as PyProgressBar


class Test_Utils(object):
    def test_progress_bar(self):
        cc_progress_bar = CcProgressBar(100, cml.losses.MSE(), cml.metrics.Accuracy())
        py_progress_bar = PyProgressBar(100, cml.losses.MSE(), cml.metrics.Accuracy())

        assert cc_progress_bar.epochs == py_progress_bar.epochs
        assert str(type(cc_progress_bar.loss)) == str(type(py_progress_bar.loss))
        assert str(type(cc_progress_bar.metric)) == str(type(py_progress_bar.metric))
