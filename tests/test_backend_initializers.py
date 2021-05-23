"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.
"""
from classicML.backend.cc.initializers import Initializer as CcInitializer
from classicML.backend.cc.initializers import RandomNormal as CcRandomNormal
from classicML.backend.cc.initializers import HeNormal as CcHeNormal
from classicML.backend.cc.initializers import XavierNormal as CcXavierNormal
from classicML.backend.cc.initializers import GlorotNormal as CcGlorotNormal
from classicML.backend.cc.initializers import RBFNormal as CcRBFNormal

from classicML.backend.python.initializers import Initializer as PyInitializer
from classicML.backend.python.initializers import RandomNormal as PyRandomNormal
from classicML.backend.python.initializers import HeNormal as PyHeNormal
from classicML.backend.python.initializers import XavierNormal as PyXavierNormal
from classicML.backend.python.initializers import GlorotNormal as PyGlorotNormal
from classicML.backend.python.initializers import RBFNormal as PyRBFNormal


class TestInitializer(object):
    def test_initializer(self):
        cc_initializer = CcInitializer()
        py_initializer = PyInitializer()

        assert cc_initializer.name == py_initializer.name
        assert cc_initializer.seed == py_initializer.seed

    def test_random_normal(self):
        cc_random_normal = CcRandomNormal(seed=2021)
        py_random_normal = PyRandomNormal(seed=2021)

        assert cc_random_normal.name == py_random_normal.name
        assert cc_random_normal.seed == py_random_normal.seed
        assert cc_random_normal(10).shape == py_random_normal(10).shape
        for key in py_random_normal([10, 10, 10]).keys():
            assert cc_random_normal([10, 10, 10])[key].shape == py_random_normal([10, 10, 10])[key].shape

    def test_he_normal(self):
        cc_he_normal = CcHeNormal(seed=2021)
        py_he_normal = PyHeNormal(seed=2021)

        assert cc_he_normal.name == py_he_normal.name
        assert cc_he_normal.seed == py_he_normal.seed
        assert cc_he_normal(10).shape == py_he_normal(10).shape
        for key in py_he_normal([10, 10, 10]).keys():
            assert cc_he_normal([10, 10, 10])[key].shape == py_he_normal([10, 10, 10])[key].shape

    def test_xavier_normal(self):
        cc_xavier_normal = CcXavierNormal(seed=2021)
        py_xavier_normal = PyXavierNormal(seed=2021)
        cc_glorot_normal = CcGlorotNormal(seed=2021)
        py_glorot_normal = PyGlorotNormal(seed=2021)

        assert cc_xavier_normal.name == py_xavier_normal.name
        assert cc_xavier_normal.seed == py_xavier_normal.seed
        assert cc_xavier_normal(10).shape == py_xavier_normal(10).shape

        assert cc_xavier_normal.name != cc_glorot_normal.name

        for key in py_xavier_normal([10, 10, 10]).keys():
            assert cc_xavier_normal([10, 10, 10])[key].shape == py_xavier_normal([10, 10, 10])[key].shape
        for key in py_glorot_normal([10, 10, 10]).keys():
            assert cc_glorot_normal([10, 10, 10])[key].shape == py_glorot_normal([10, 10, 10])[key].shape

        for key in cc_xavier_normal([10, 10, 10]).keys():
            assert (cc_xavier_normal([10, 10, 10])[key] == cc_glorot_normal([10, 10, 10])[key]).all()

    def test_rbf_normal(self):
        cc_rbf_normal = CcRBFNormal(seed=2021)
        py_rbf_normal = PyRBFNormal(seed=2021)

        assert cc_rbf_normal.name == py_rbf_normal.name
        assert cc_rbf_normal.seed == py_rbf_normal.seed
        assert cc_rbf_normal(10)['w'].shape == py_rbf_normal(10)['w'].shape
        assert cc_rbf_normal(10)['b'].shape == py_rbf_normal(10)['b'].shape
        assert cc_rbf_normal(10)['c'].shape == py_rbf_normal(10)['c'].shape
        assert cc_rbf_normal(10)['beta'].shape == py_rbf_normal(10)['beta'].shape
