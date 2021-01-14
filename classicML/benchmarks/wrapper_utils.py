import os
import time

import psutil

from classicML import CLASSICML_LOGGER


def memory_monitor(function):
    """内存监视装饰器.

    Notes:
        - 使用该函数统计内存信息, 有潜在降低运行速度的可能性.
        并且psutil针对的Python优化手段会导致在CC引擎的速度大幅降低.
    """
    def wrapper(*args, **kwargs):
        return_values = function(*args, **kwargs)

        pid = os.getpid()
        current_process = psutil.Process(pid)
        process_memory = current_process.memory_full_info()
        CLASSICML_LOGGER.info('占用内存 {:.5f} MB'.format(process_memory.uss / 1024 / 1024))
        return return_values

    return wrapper


def timer(function):
    """程序计时装饰器.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 注意将记录time.sleep()的时间
        return_values = function(*args, **kwargs)
        end_time = time.perf_counter()
        CLASSICML_LOGGER.info('耗时 {:.5f} s'.format(end_time - start_time))

        return return_values

    return wrapper