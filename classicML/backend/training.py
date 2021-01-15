"""classicML的training模块用于安全的智能的访问classicML的后端函数."""
from classicML import CLASSICML_LOGGER
from classicML.backend import initializers
from classicML.backend import kernels
from classicML.backend import losses
from classicML.backend import optimizers
from classicML.backend import metrics
from classicML.backend.python import tree


def get_criterion(criterion):
    """获取使用的划分选择方式.

    Arguments:
        criterion: str,
            决策树学习的划分方式.

    Raises:
        AttributeError: 选择错误.
    """
    if criterion == 'gini':
        return tree.criteria.Gini()
    elif criterion == 'gain':
        return tree.criteria.Gain()
    elif criterion == 'entropy':
        return tree.criteria.Entropy()
    else:
        CLASSICML_LOGGER.error('选择错误')
        raise AttributeError


def get_initializer(initializer, seed):
    """获取使用的初始化器实例.

    Arguments:
        initializer: str or classicML.initializers.Initializer 实例,
            初始化器.
        seed: int, 初始化器的随机种子.

    Raises:
        AttributeError: 模型编译的参数输入错误.
    """
    if isinstance(initializer, str):
        if initializer == 'random_normal':
            return initializers.RandomNormal(seed=seed)
        elif initializer == 'he_normal':
            return initializers.HeNormal(seed=seed)
        elif initializer in ('xavier_normal', 'glorot_normal'):
            return initializers.XavierNormal(seed=seed)
        elif initializer == 'rbf_normal':
            return initializers.RBFNormal(seed=seed)
        else:
            CLASSICML_LOGGER.error('初始化器调用错误')
            raise AttributeError
    elif isinstance(initializer, initializers.Initializer):
        return initializer
    elif initializer is None:
        return initializers.RandomNormal(seed=seed)
    else:
        CLASSICML_LOGGER.error('初始化器调用错误')
        raise AttributeError


def get_kernel(kernel, gamma):
    """获取使用的核函数实例.

    Arguments:
        kernel: str or classicML.kernels.Kernel 实例,
            核函数.
        gamma: float, 核函数系数.

    Raises:
        AttributeError: 模型编译的参数输入错误.
    """
    if isinstance(kernel, str):
        if kernel == 'linear':
            return kernels.Linear()
        elif kernel == 'rbf':
            return kernels.RBF(gamma=gamma)
        elif kernel == 'gaussian':
            return kernels.Gaussian(gamma=gamma)
        elif kernel == 'poly':
            return kernels.Polynomial(gamma=gamma)
        elif kernel == 'sigmoid':
            return kernels.Sigmoid(gamma=gamma)
        else:
            CLASSICML_LOGGER.error('核函数调用错误')
            raise AttributeError
    elif isinstance(kernel, kernels.Kernel):
        return kernel
    else:
        CLASSICML_LOGGER.error('核函数调用错误')
        raise AttributeError


def get_loss(loss):
    """获取使用的损失函数实例.

    Arguments:
        loss: str or classicML.losses.Loss 实例,
            损失函数.
    """
    if isinstance(loss, str):
        if loss in ('mse', 'mean_squared_error'):
            return losses.MeanSquaredError()
        elif loss == 'log_likelihood':
            return losses.LogLikelihood()
        elif loss == 'binary_crossentropy':
            return losses.BinaryCrossentropy()
        elif loss == 'categorical_crossentropy':
            return losses.CategoricalCrossentropy()
        elif loss == 'crossentropy':
            return losses.Crossentropy()
        else:
            CLASSICML_LOGGER.warn('你没有输入损失函数或者输入的损失函数不正确, 将使用默认的损失函数')
            return losses.Crossentropy()
    elif isinstance(loss, losses.Loss):
        return loss
    else:
        CLASSICML_LOGGER.warn('你没有输入损失函数或者输入的损失函数不正确, 将使用默认的损失函数')
        return losses.Crossentropy()


def get_metric(metric):
    """获取使用的评估函数实例.

    Arguments:
        metric: str or classicML.metrics.Metric 实例,
            评估函数.

    Raises:
        AttributeError: 模型编译的参数输入错误.
    """
    if isinstance(metric, str):
        if metric == 'binary_accuracy':
            return metrics.BinaryAccuracy()
        elif metric == 'categorical_accuracy':
            return metrics.CategoricalAccuracy()
        elif metric == 'accuracy':
            return metrics.Accuracy()
    elif isinstance(metric, metrics.Metric):
        return metric
    else:
        CLASSICML_LOGGER.error('评估函数调用错误')
        raise AttributeError


def get_optimizer(optimizer):
    """获取使用的优化器实例.

    Arguments:
        optimizer: str or classicML.optimizers.Optimizer 实例,
            优化器.

    Raises:
        AttributeError: 模型编译的参数输入错误.
    """
    if isinstance(optimizer, str):
        if optimizer in ('gd', 'gradient_descent'):
            return optimizers.GradientDescent()
        elif optimizer in ('newton', 'newton_method'):
            return optimizers.NewtonMethod()
        elif optimizer in ('sgd', 'stochastic_gradient_descent'):
            return optimizers.StochasticGradientDescent()
        elif optimizer == 'adam':
            return optimizers.Adam()
        elif optimizer == 'rbf':
            return optimizers.RadialBasisFunctionOptimizer()
        elif optimizer == 'SMO':
            return optimizers.SequentialMinimalOptimization()
        else:
            CLASSICML_LOGGER.error('优化器调用错误')
            raise AttributeError
    elif isinstance(optimizer, optimizers.Optimizer):
        return optimizer
    else:
        CLASSICML_LOGGER.error('优化器调用错误')
        raise AttributeError


def get_pruner(pruning):
    """获取剪枝器.

    Arguments:
        pruning: str,
            决策树剪枝的方式.

    Raises:
        AttributeError: 选择错误.
    """
    if pruning == 'post':
        return tree.pruners.PostPruner()
    elif pruning == 'pre':
        return tree.pruners.PrePruner()
    elif pruning is None:
        return None
    else:
        CLASSICML_LOGGER.error('选择错误')
        raise AttributeError