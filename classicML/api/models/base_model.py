from abc import ABC
from abc import abstractmethod

from classicML.backend import io


class BaseModel(ABC):
    """classicML的模型抽象基类, classicML的模型全部继承于此.
    通过继承BaseModel, 至少实现fit和predict方法就可以构建自己的模型.
    """
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def compile(self, **kwargs):
        """编译朴素贝叶斯分类器."""
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        """训练模型.

        Arguments:
            x: array-like, 特征数据
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

    def load_weights(self, filepath):
        """加载模型参数,
        使用parameters_gp加载保存的参数.

        Arguments:
            filepath: str, 权重文件加载的路径.

        Raises:
            NotImplemented: 需要用户自行实现.
        """
        parameters_gp = io.initialize_weights_file(filepath,
                                                   mode='r',
                                                   model_name='BaseModel')
        raise NotImplemented

    def save_weights(self, filepath):
        """将模型权重保存为一个HDF5文件,
        使用parameters_gp保存的参数.

        Arguments:
            filepath: str, 权重文件保存的路径.

        Raises:
            NotImplemented: 需要用户自行实现.
        """
        parameters_gp = io.initialize_weights_file(filepath,
                                                   mode='w',
                                                   model_name='BaseModel')

        raise NotImplemented
