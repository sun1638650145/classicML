"""
    测试backend.io模块的写入和读取是否正常.
"""
import os

import numpy as np

from classicML.backend import io


class TestInitializeWeightsFile(object):
    def test_answer(self):
        filepath = './test.h5'
        # 数据为随机产生, 不具有任何实际意义.
        _temp_arr = np.random.rand(3, 4)

        # 创建并写入
        w_parameters_gp = io.initialize_weights_file(filepath, mode='w', model_name='test')
        w_compile_ds = w_parameters_gp['compile']
        w_weights_ds = w_parameters_gp['weights']

        w_compile_ds.attrs['module'] = 'classicML'
        w_compile_ds.attrs['cml_version'] = io.cml_version
        w_compile_ds.attrs['__version__'] = io.__version__
        w_weights_ds.attrs['weights'] = _temp_arr

        # 读取并核验
        r_parameters_gp = io.initialize_weights_file(filepath, mode='r', model_name='test')
        r_compile_ds = r_parameters_gp['compile']
        r_weights_ds = r_parameters_gp['weights']

        assert 'classicML' == r_compile_ds.attrs['module']
        assert io.cml_version == r_compile_ds.attrs['cml_version']
        assert io.__version__ == r_compile_ds.attrs['__version__']
        assert _temp_arr.all() == r_weights_ds.attrs['weights'].all()

        if os.path.exists(filepath):
            os.remove(filepath)