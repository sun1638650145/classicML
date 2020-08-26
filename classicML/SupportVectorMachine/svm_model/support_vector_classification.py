import numpy as np
from .optimizer import SMO, kappa_xi_x


class SupportVectorClassification:
    """支持向量分类器"""
    def __init__(self, seed=None):
        """
        支持向量分类器初始化
        Parameters
        ----------
        seed : int or None, default=None, optional
            随机种子
        """
        np.random.seed(seed)

        self.support_ = None
        self.support_vector_ = None
        self.support_alpha_ = None
        self.support_y_ = None

    def compile(self, C=10000.0, kernel='rbf', gamma='auto', max_iter=-1, degree=3, tol=1e-3):
        """
        给支持向量分类器进行编译,配置参数项
        Parameters
        ----------
        C : float, optional
            软间隔的正则化系数
        kernel : str, optional
            核函数, 默认是高斯核函数
            现在支持的核函数有高斯核'rbf', 线性核'linear', 多项式核'poly'
        gamma : {'auto', 'scale'} or float, optional
            核函数系数
        max_iter : int, default=-1, optional
            最大的迭代次数(可避免过拟合)
            -1表示没有限制, 直到找到所有样本都满足KKT条件的时候
        degree : int, default=3, optional
            多项式核函数的次数
            如果不使用多项式核函数, 此参数无效, 默认3
        tol : float, default=1e-3, optional
            停止训练的误差值
        """
        assert kernel in ('rbf', 'linear', 'poly')
        assert gamma in ('auto', 'scale') or type(gamma) == float or type(gamma) == int

        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree
        self.tol = tol
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
            kappa_i = kappa_xi_x(self.kernel, x[sample], self.support_vector_, self.degree, self.gamma)
            fx = np.dot((self.support_alpha_.reshape(-1, 1) * self.support_y_).T, kappa_i) + self.b
            if fx < 0:
                y_pred[sample] = -1

        return y_pred


# alias
SVC = SupportVectorClassification