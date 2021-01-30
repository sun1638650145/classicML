class History(object):
    """保存训练的历史记录.

    Attributes:
        name: str, default=None,
            历史记录的名称.
        loss_name: str, default='loss'
            使用损失函数的名称.
        metric_name: str, default='metric'
            使用评估函数的名称.
        loss: list, 损失值组成的列表
        metric: list, 评估值组成的列表

    Notes:
        - 使用该函数记录数据, 会导致运行速度的降低.
    """
    def __init__(self, name=None, loss_name='loss', metric_name='metric'):
        """
        Arguments:
            name: str, default=None,
                历史记录的名称.
            loss_name: str, default='loss'
                使用损失函数的名称.
            metric_name: str, default='metric'
                使用评估函数的名称.
        """
        self.name = name
        self.loss_name = loss_name
        self.metric_name = metric_name

        self.loss = []
        self.metric = []

    def __call__(self, loss_value, metric_value):
        """记录当前的信息.

        Arguments:
            loss_value: float, 当前的损失值.
            metric_value: float, 当前的评估值.
        """
        self.loss.append(loss_value)
        self.metric.append(metric_value)
