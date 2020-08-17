import numpy as np
from time import time, sleep


def display_verbose(epoch, epochs, loss, accuracy, starting_time, ETD, timesleep=0):
    """进度条显示函数"""
    if epoch == 0:
        print('\rEpoch %d/%d [>........................] ETA: 00:00 - loss: %.4f - accuracy: %.4f' % (epoch + 1, epochs, loss, accuracy), end='')
    elif epoch > (epochs - epochs / 25):
        total_time = time() - ETD
        time_per_epoch = total_time * 1000 / epochs
        print('\rEpoch %d/%d [=========================] %.0fs %.0fms/step - loss: %.4f - accuracy: %.4f' % (epoch + 1, epochs, total_time, time_per_epoch, loss, accuracy), end='')
    else:
        print('\rEpoch %d/%d [' % (epoch + 1, epochs), end='')

        # 箭头位置
        arrow = int(np.ceil((epoch + 1) / (epochs / 25)))
        for num in range(arrow-1):
            print('=', end='')
        print('>', end='')
        for num in range(25 - arrow):
            print('.', end='')

        # 预计时间
        epoch_time = time() - starting_time
        ETA = epoch_time * (epochs - epoch - 1)
        if ETA < 60:
            print('] ETA: %.0fs - loss: %.4f - accuracy: %.4f' % (ETA, loss, accuracy), end='')
        else:
            ETA_minutes = int(ETA) / 60
            ETA_seconds = int(ETA) % 60
            print('] ETA: %02d:%02d - loss: %.4f - accuracy: %.4f' % (ETA_minutes, ETA_seconds, loss, accuracy), end='')
    sleep(timesleep)