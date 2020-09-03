import numpy as np
from .optimizer import SMO
from .backend import kappa


class SupportVectorClassification:
    """支持向量分类器"""
    def __init__(self, seed=None):
        """
        支持向量分类器初始化
        Parameters
        ----------
        seed : int, default=None, optional
            随机种子

        Arguments
        ---------
        support_ : numpy.ndarray
            支持向量分类器的支持向量下标组成的数组
        support_vector_ : numpy.ndarray
            支持向量分类器的支持向量组成的数组
        support_alpha_ : numpy.ndarray
            支持向量分类器的拉格朗日乘子组成的数组
        support_y_ : numpy.ndarray
            支持向量分类器的支持向量对应的标签组成的数组
        """
        np.random.seed(seed)

        self.support_ = None
        self.support_vector_ = None
        self.support_alpha_ = None
        self.support_y_ = None

    def compile(self, C=10000.0, kernel='rbf', gamma='auto', max_iter=-1, degree=3, beta=1., theta=-1., tol=1e-3, customize_kernel=None):
        """
        给支持向量分类器进行编译,配置参数项
        Parameters
        ----------
        C : float, optional
            软间隔的正则化系数
        kernel : {'rbf', 'linear', 'poly', 'sigmoid', 'customize'}, optional
            核函数, 默认是高斯核函数
            注意, 使用'customize'的时候, customize_kernel需要提供
        gamma : {'auto', 'scale'} or float, optional
            高斯核函数系数
            如果不使用高斯核函数, 此参数无效
        max_iter : int, default=-1, optional
            最大的迭代次数(可避免过拟合)
            -1表示没有限制, 直到找到所有样本都满足KKT条件的时候
        degree : int, default=3, optional
            多项式核函数的次数
            如果不使用多项式核函数, 此参数无效
        beta : float, default=1, optional
            sigmoid核函数的参数
            如果不使用sigmoid核函数, 此参数无效
        theta : float, default=-1, optional
            sigmoid核函数的参数, 小于0的整数
            如果不使用sigmoid核函数, 此参数无效
        tol : float, default=1e-3, optional
            停止训练的误差值
        customize_kernel : function, default=None, optional
            自定义核函数
            kernel是'customize', 需要用户手动实现核函数, 核函数的形式必须是my_kernel(x_i, x_j); kernel不是'customize', 该参数无效
            注意, 自定义的核函数不一定保证存在解
        """
        assert kernel in ('rbf', 'linear', 'poly', 'sigmoid', 'customize')
        assert gamma in ('auto', 'scale') or type(gamma) == float or type(gamma) == int

        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree
        self.beta = beta
        self.theta = theta
        self.tol = tol
        self.customize_kernel = customize_kernel
        if type(gamma) is int:
            self.gamma = float(gamma)
        else:
            self.gamma = gamma

    def fit(self, x, y):
        """
        给支持向量分类器进行训练
        Parameters
        ----------
        x : numpy.ndarray or array_like
            特征数据
        y : numpy.ndarray or array_like
            标签
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.gamma == 'auto':
            self.gamma = 1.0 / x.shape[1]
        elif self.gamma == 'scale':
            self.gamma = 1.0 / (x.shape[1] * x.var())
        else:
            pass

        self.b = 0
        self.alphas = np.zeros((x.shape[0],))
        self.non_bound_alpha = np.zeros((x.shape[0],))
        self.error_cache = np.zeros((x.shape[0],))
        self.non_zero_alpha = np.zeros((x.shape[0],), dtype=bool)

        # 使用SMO算法优化
        SMO(self, x, y)

        self.support_ = self.non_zero_alpha.nonzero()[0]
        self.support_vector_ = x[self.support_]
        self.support_alpha_ = self.alphas[self.non_zero_alpha]
        self.support_y_ = y[self.support_]

        return self

    def predict(self, x):
        """
        进行预测
        Parameters
        ----------
        x : numpy.ndarray or array_like
            特征数据

        Returns
        -------
        y_pred : numpy.ndarray
            支持向量分类器预测的标签数组
        """
        if self.support_vector_ is None:
            raise Exception('你必须先进行训练')

        num_of_sample = x.shape[0]
        y_pred = np.ones((num_of_sample, ))
        for sample in range(num_of_sample):
            kappa_i = kappa(self.kernel, x[sample], self.support_vector_, self.degree, self.gamma, self.beta, self.theta, self.customize_kernel)
            fx = np.dot((self.support_alpha_.reshape(-1, 1) * self.support_y_).T, kappa_i) + self.b
            if fx < 0:
                y_pred[sample] = -1

        return y_pred


# alias
SVC = SupportVectorClassification