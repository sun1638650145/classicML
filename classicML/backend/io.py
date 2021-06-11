"""classicML的I/O函数, 用于模型保存等操作."""
import re
from time import time

import h5py
import numpy as np

from classicML import CLASSICML_LOGGER
from classicML import __version__ as cml_version

__version__ = 'backend.io.0.5.1'

min_cml_version = '0.5'
min__version__ = 'backend.io.0.3'


def _parse(model_name, fp):
    """解析描述信息组, 核验文件.

    Arguments:
        model_name: str, 模型的名称.
        fp: h5py._hl.files.File, 文件指针.

    """
    description_gp = fp['description']

    file_cml_version = re.findall('\\d+\\.\\d+(?:\\.\\d+)*', description_gp.attrs['version'])[0]
    file_backend_version = 'backend.io.' + re.findall('\\d+\\.\\d+', description_gp.attrs['version'])[-1]

    if (file_cml_version < min_cml_version) or (file_backend_version < min__version__):
        CLASSICML_LOGGER.error('文件核验失败, 模型版本过低')
        raise ValueError('文件核验失败')
    if description_gp.attrs['model_name'] != model_name:
        CLASSICML_LOGGER.error('文件核验失败, 模型不匹配')
        raise ValueError('文件核验失败')


def initialize_weights_file(filepath, mode, model_name):
    """初始化权重文件, 以创建或者解析模式运行.

    cml.backend.io的HDF5文件标准化协议包括:
      两个信息组description和parameters,
      description用来存放cml兼容性和开发时间等信息; parameters用以保存模型本身的参数.
      parameters分成两个数据集:
        compile保存模型的训练的超参数; weights保存模型的权重信息.
    开发符合标准化协议的自定义模型, 需将固化的参数保存在compile和weights中.

    具体结构如下:
      /
      ｜-- description
          |-- version 版本信息
          |-- model_name 模型名称
          |__ saved_time 时间戳
      ｜__ parameters
          ｜-- compile 训练相关超参数
          ｜__ weights 模型的权重

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
        if mode == 'w':
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
            _parse(model_name, fp)
            # 提取参数信息组.
            parameters_gp = fp['parameters']
    except IOError:
        CLASSICML_LOGGER.error('模型权重文件初始化失败, 请检查文件')
        raise IOError('初始化失败')

    return parameters_gp
