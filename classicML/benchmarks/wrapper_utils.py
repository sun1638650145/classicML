import os
import time

import numpy as np
import psutil

from classicML import CLASSICML_LOGGER


# TODO(Steve R. Sun, tag:code): 将@timer和@average_timer()合并, 并且可以不使用括号.
def average_timer(repeat=5):
    """程序平均计时装饰器.

    Arguments:
        repeat: int, default=5,
            重复运行的次数.

    Notes:
        - 使用该装饰器统计平均计时会明显降低运行速度,
        请在开发时使用, 避免在训练模型时使用.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            return_values = None
            average_time = list()

            for i in range(repeat):
                start_time = time.perf_counter()
                return_values = function(*args, **kwargs)
                end_time = time.perf_counter()

                average_time.append(end_time - start_time)

            CLASSICML_LOGGER.info('平均耗时 {:.5f} ± {:.5f} s(mean ± std. 循环次数 {:d})'
                                  .format(np.mean(average_time), np.std(average_time), repeat))

            # 函数返回值为最后一次的返回值.
            return return_values

        return wrapper

    return decorator


def memory_monitor(function):
    """内存监视装饰器.

    Notes:
        - 使用该装饰器统计内存信息, 有潜在降低运行速度的可能性.
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
