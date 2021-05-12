"""classicML的评估函数."""
import numpy as np


class Metric(object):
    """评估函数的基类.

    Attributes:
        name: str, default='metric',
            评估函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    def __init__(self, name='metric'):
        """
        Arguments:
            name: str, default='metric',
                评估函数名称.
        """
        self.name = name

    def __call__(self, y_pred, y_true):
        """函数实现."""
        raise NotImplementedError


class Accuracy(Metric):
    """准确率评估函数,
    将根据标签的实际形状自动使用二分类或者多分类评估函数.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    def __init__(self, name='accuracy'):
        super(Accuracy, self).__init__(name=name)

    def __call__(self, y_pred, y_true):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的准确率.
        """
        if y_pred.shape[1] == 1:
            metric = BinaryAccuracy()
        else:
            metric = CategoricalAccuracy()
        accuracy = metric(y_pred, y_true)

        return accuracy


class BinaryAccuracy(Metric):
    """二分类准确率评估函数.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    def __init__(self, name='binary_accuracy'):
        super(BinaryAccuracy, self).__init__(name=name)

    def __call__(self, y_pred, y_true):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的准确率.
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        accuracy = np.mean(np.equal(y_pred, y_true))

        return accuracy


class CategoricalAccuracy(Metric):
    """多分类准确率评估函数.

    Notes:
        - 该函数提供了非Python后端的实现版本,
          你可以使用其他的版本, 函数的调用方式和接口一致,
          Python版本是没有优化的原始公式版本.
    """
    def __init__(self, name='categorical_accuracy'):
        super(CategoricalAccuracy, self).__init__(name=name)

    def __call__(self, y_pred, y_true):
        """函数实现.

        Arguments:
            y_pred: numpy.ndarray, 预测的标签.
            y_true: numpy.ndarray, 真实的标签.

        Returns:
            当前的准确率.
        """
        y_pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        y_true = [np.argmax(y_true[i]) for i in range(len(y_true))]

        accuracy = np.mean(np.equal(y_pred, y_true))

        return accuracy
