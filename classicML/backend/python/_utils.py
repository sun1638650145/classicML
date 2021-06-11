"""classicML的工具类."""
import sys
from time import time

import numpy as np


class ProgressBar(object):
    """训练进度条.

    Attributes:
        ETD: float, 优化器启动的时间戳.
        epochs: int, 训练的轮数.
        loss: str, classicML.losses.Loss 实例,
            模型使用的损失函数.
        metric: str, classicML.metrics.Metric 实例,
            模型使用的评估函数.
    """

    def __init__(self, epochs, loss, metric):
        """
        Arguments:
            epochs: int, 训练的轮数.
            loss: str, classicML.losses.Loss 实例,
                模型使用的损失函数.
            metric: str, classicML.metrics.Metric 实例,
                模型使用的评估函数.
        """
        super(ProgressBar, self).__init__()
        self.ETD = time()
        self.epochs = epochs
        self.loss = loss
        self.metric = metric

    def __call__(self, epoch, current, loss_value, metric_value):
        """函数实现.

        Arguments:
            epoch: int, 当前的训练轮数.
            current: float, 当前的时间戳.
            loss_value: float, 当前的损失值.
            metric_value: float, 当前的评估值.
        """
        self._update_info(epoch, current, loss_value, metric_value)
        self._dynamic_display()

    def _update_info(self, epoch, current, loss_value, metric_value):
        """在终端上显示进度条.

        Arguments:
            epoch: int, 当前的训练轮数.
            current: float, 当前的时间戳.
            loss_value: float, 当前的损失值.
            metric_value: float, 当前的评估值.
        """
        self._draw_bar(epoch)
        self._draw_detail(epoch, current, loss_value, metric_value)

    def _dynamic_display(self):
        """在终端显示.
        """
        sys.stdout.write('\r')
        sys.stdout.write(self.info)
        sys.stdout.flush()

    def _draw_bar(self, epoch):
        """绘制进度条.
        无论总的训练轮数是多少, 显示条的总更新步数是25次.

        Arguments:
            epoch: int, 当前的训练轮数.
        """
        self.info = 'Epoch {}/{} ['.format(epoch, self.epochs)

        if epoch > (self.epochs - self.epochs / 25):
            self.info += '=========================]'
        else:
            # 获取箭头的实时位置
            # arrow = epoch / (epochs / 25)
            arrow = int(np.ceil(epoch / self.epochs * 25))

            self.info += '=' * (arrow - 1)
            self.info += '>'
            self.info += '.' * (25 - arrow)
            self.info += ']'

    def _draw_detail(self, epoch, current, loss_value, metric_value):
        """绘制显示的计算信息.

        Arguments:
            epoch: int, 当前的训练轮数.
            current: float, 当前的时间戳.
            loss_value: float, 当前的损失值.
            metric_value: float, 当前的评估值.
        """
        # 时间信息
        if epoch == 0:
            self.info += ' ETA: 00:00'
        elif epoch > (self.epochs - self.epochs / 25):
            self.total = time() - self.ETD
            self.per_epoch = self.total * 1000 / self.epochs
            self.info += ' {:.0f}s {:.0f}ms/step'.format(self.total, self.per_epoch)
        else:
            ETA = (time() - current) * (self.epochs - epoch)  # 剩余时间
            if ETA < 60:
                self.info += ' ETA: {:.0f}s'.format(ETA)
            else:
                ETA_minutes = int(ETA) // 60
                ETA_seconds = int(ETA) % 60
                self.info += ' ETA: {:02d}:{:02d}'.format(ETA_minutes, ETA_seconds)

        # 数值信息
        self.info += ' - {}: {:.4f} - {}: {:.4f}'.format(self.loss.name, loss_value, self.metric.name, metric_value)

        if epoch == self.epochs:
            self.info += '\n'
