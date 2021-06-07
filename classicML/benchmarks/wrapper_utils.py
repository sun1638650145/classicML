import os
import time

import numpy as np
import psutil

from classicML import CLASSICML_LOGGER


def _format_and_display_the_time_spent(start_time=None, end_time=None, time_spent_list=None, repeat=None):
    """格式化并显示运行时间.

    Arguments:
        start_time: float, default=None,
            函数开始运行的时间.
        end_time: float, default=None,
            函数结束的时间.
        time_spent_list: numpy.ndarray, default=None,
            记录多次运行时间的列表.
        repeat: int, default=None,
            重复运行的次数.
    """
    if start_time is not None:
        time_spent = (end_time - start_time) * 1000 * 1000  # 返回时间的单位是s, s -> us.

        if int(time_spent) // 1000 // 1000 > 0:
            CLASSICML_LOGGER.info('耗时 {:.3f} s'.format(time_spent / 1000 / 1000))
        elif int(time_spent) // 1000 > 0:
            CLASSICML_LOGGER.info('耗时 {:.3f} ms'.format(time_spent / 1000))
        else:
            CLASSICML_LOGGER.info('耗时 {:.3f} us'.format(time_spent))
    else:
        time_spent_list = time_spent_list * 1000 * 1000  # 返回时间的单位是s, s -> us.

        average_time_spent = np.mean(time_spent_list)
        std_time = np.std(time_spent_list)
        min_time = np.min(time_spent_list)
        max_time = np.max(time_spent_list)

        if int(max_time) // 1000 // 1000 > 0:
            average_time_spent /= (1000 * 1000)
            std_time /= 1000
            min_time /= (1000 * 1000)
            max_time /= (1000 * 1000)
            unit1, unit2 = 's', 'ms'
        elif int(max_time) // 1000 > 0:
            average_time_spent /= 1000
            min_time /= 1000
            max_time /= 1000
            unit1, unit2 = 'ms', 'us'
        else:
            unit1, unit2 = 'us', 'us'

        CLASSICML_LOGGER.info('平均耗时 {:.3f} %s ± {:.0f} %s, {:.3f} %s, {:.3f} %s; 循环次数 {:d} (mean ± std, max, min)'
                              .format(average_time_spent, std_time, max_time, min_time, repeat)
                              % (unit1, unit2, unit1, unit1))


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
            time_spent_list = list()

            for i in range(repeat):
                start_time = time.perf_counter()
                return_values = function(*args, **kwargs)
                end_time = time.perf_counter()

                time_spent_list.append(end_time - start_time)

            _format_and_display_the_time_spent(None, None, np.asarray(time_spent_list), repeat)

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

        _format_and_display_the_time_spent(start_time, end_time, None, None)

        return return_values

    return wrapper
