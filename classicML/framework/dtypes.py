"""classicML的数据类型. 在v0.7中引入精度控制, 预定义了float32, int32, float64, int64; 4种类型完全继承自对应的numpy的数据类型.
"""
import os

import numpy as np

from classicML import CLASSICML_LOGGER


class float32(np.float32):
    """32位浮点数."""
    pass


class float64(np.float64):
    """64位浮点数."""
    pass


class int32(np.int32):
    """32位整数."""
    pass


class int64(np.int64):
    """64位整数."""
    pass


def set_precision():
    """设置全局的精度."""
    class _precision(object):
        float = None
        int = None

    if os.environ['CLASSICML_PRECISION'] == '32-bit':
        _precision.float = float32
        _precision.int = int32
    elif os.environ['CLASSICML_PRECISION'] == '64-bit':
        _precision.float = float64
        _precision.int = int64
    else:
        CLASSICML_LOGGER.warn('您预设的精度不能正常识别, 将精度自动设置为默认的32位')
        _precision.float = float32
        _precision.int = int32

    return _precision
