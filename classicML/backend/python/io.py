from time import time

import h5py
import numpy as np

from classicML import CLASSICML_LOGGER
from classicML import __version__ as cml_version

__version__ = 'backend.io.0.3'


def initialize_weights_file(filepath, mode, model_name):
    """初始化权重文件, 以创建或者解析模式运行.

    Arguments:
        filepath: str, 权重文件加载的路径.
        mode: {'w', 'r'},
            工作模式, 'w'是写入权重文件, 'r'是读取权重文件.
        model_name: str, 模型的名称.

    Returns:
        可操作的文件指针.

    Raises:
        IOError: 初始化失败.
        ValueError: 文件核验失败.
    """
    try:
        fp = h5py.File(filepath, mode)
        if mode is 'w':
            # 创建描述信息组.
            description_gp = fp.create_group(name='description')
            description_gp.attrs['version'] = cml_version + '.' + __version__
            description_gp.attrs['model_name'] = model_name
            description_gp.attrs['saved_time'] = time()

            # 创建参数信息组(包括两个数据集, 分别记录超参数和模型权重).
            parameters_gp = fp.create_group(name='parameters')
            parameters_gp.create_dataset(name='compile', dtype=np.float64)
            parameters_gp.create_dataset(name='weights', dtype=np.float64)
        else:
            # 解析描述信息组.
            description_gp = fp['description']
            if description_gp.attrs['version'] != cml_version + '.' + __version__:
                CLASSICML_LOGGER.error('文件核验失败, 版本不兼容')
                raise ValueError('文件核验失败')
            if description_gp.attrs['model_name'] != model_name:
                CLASSICML_LOGGER.error('文件核验失败, 模型不匹配')
                raise ValueError('文件核验失败')

            # 提取参数信息组.
            parameters_gp = fp['parameters']
    except IOError:
        CLASSICML_LOGGER.error('模型权重文件初始化失败, 请检查文件')
        raise IOError('初始化失败')

    return parameters_gp