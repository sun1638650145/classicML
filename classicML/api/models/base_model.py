from abc import ABC
from abc import abstractmethod

from classicML.backend import metrics


class BaseModel(ABC):
    """classicML的模型抽象基类, classicML的模型全部继承于此.
    通过继承BaseModel, 至少实现fit和predict方法就可以构建自己的模型.
    """
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def compile(self, **kwargs):
        """编译模型."""
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        """训练模型.

        Arguments:
            x: array-like, 特征数据.
            y: array-like, 标签.

        Raises:
            NotImplementedError: 需要用户自行实现.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x, **kwargs):
        """使用模型进行预测.

        Arguments:
            x: array-like, 特征数据.

        Raises:
            NotImplementedError: 需要用户自行实现.
        """
        raise NotImplementedError

    def score(self, x, y):
        """在预测模式下计算准确率.

        Arguments:
            x: array-like, 特征数据.
            y: array-like, 标签.

        Returns:
            当前的准确率.
        """
        y_pred = self.predict(x)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        accuracy = metrics.Accuracy()(y_pred, y)

        return accuracy

    def load_weights(self, filepath):
        """加载模型参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            NotImplemented: 需要用户自行实现.
        """
        raise NotImplemented

    def save_weights(self, filepath):
        """将模型权重保存,
         如果您希望您的模型参数收到保护, 可自行实现模型的保存方式;
         如果您希望您的模型开源, 请参照cml.backend.io的协议方式实现参数的保存.

        Arguments:
            filepath: str, 权重文件保存的路径.

        Raises:
            NotImplemented: 需要用户自行实现.
        """
        raise NotImplemented
