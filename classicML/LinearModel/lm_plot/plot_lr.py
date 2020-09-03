import numpy as np
import matplotlib.pyplot as plt
# 设置中文
plt.rcParams['font.family'] = 'Arial Unicode MS'


def plot_logistic_regression(beta, x, y):
    """可视化逻辑回归"""
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[1] != 2:
        raise Exception('目前暂时不能可视化非二维数据')

    num_of_sample = len(y)
    positive = len(y[y == 1])

    x_crood = np.linspace(0, 1)
    # 0 = x1 * beta[0] + x2 * beta[1] + beta[2]
    y_crood = -(beta[0] * x_crood + beta[2]) / beta[1]
    plt.plot(x_crood, y_crood, c='orange', label='逻辑回归-logistic regression')

    plt.scatter(x[:, 0][: positive], x[:, 1][: positive], label='正例', c='c', marker='o')
    plt.scatter(x[:, 0][positive: num_of_sample], x[:, 1][positive: num_of_sample], label='反例', c='lightcoral', marker='o')

    plt.legend()
    plt.show()