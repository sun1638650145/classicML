"""
    测试backend.io模块的写入和读取是否正常.
"""
import os

from classicML.backend import io


class TestInitializeWeightsFile(object):
    def test_answer(self):
        filepath = './test.h5'

        w_parameters_ds = io.initialize_weights_file(filepath, mode='w', model_name='test')
        w_parameters_ds.attrs['module'] = 'classicML'
        w_parameters_ds.attrs['cml_version'] = io.cml_version
        w_parameters_ds.attrs['__version__'] = io.__version__
        r_parameters_ds = io.initialize_weights_file(filepath, mode='r', model_name='test')

        assert 'classicML' == r_parameters_ds.attrs['module']
        assert io.cml_version == r_parameters_ds.attrs['cml_version']
        assert io.__version__ == r_parameters_ds.attrs['__version__']

        if os.path.exists(filepath):
            os.remove(filepath)