import logging

import numpy as np

from .backend import get_within_class_scatter_matrix, get_w

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='classicML')


class LinearDiscriminantAnalysis:
    """线性判别分析"""
    def __init__(self):
        self.mu_0 = None
        self.mu_1 = None
        self.w = None

    def fit(self, x, y):
        """训练"""
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(int)

        X_0 = x[y == 0]
        X_1 = x[y == 1]

        self.mu_0 = np.mean(X_0, axis=0, keepdims=True)
        self.mu_1 = np.mean(X_1, axis=0, keepdims=True)

        S_w = get_within_class_scatter_matrix(X_0, X_1, self.mu_0, self.mu_1)
        self.w = get_w(S_w, self.mu_0, self.mu_1)

        return self

    def predict(self, x):
        """预测"""
        if self.w is None:
            logger.error('你必须先进行训练')
            raise ValueError

        crood = np.dot(x, self.w.T)
        wu_0 = np.dot(self.w, self.mu_0.T)
        wu_1 = np.dot(self.w, self.mu_1.T)

        y_pred = np.squeeze((np.abs(crood - wu_0) > np.abs(crood - wu_1)))

        return y_pred.astype(int)