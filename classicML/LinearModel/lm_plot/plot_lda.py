import numpy as np
import matplotlib.pyplot as plt

# 设置中文
plt.rcParams['font.family'] = 'Arial Unicode MS'


def set_ax(ax):
    # 设置坐标轴
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')


def plot_linear_discriminant_analysis(model, x, y,  xlabel=None, ylabel=None):
    """可视化线性判别分析"""
    w, mu_0, mu_1 = model.w, model.mu_0, model.mu_1

    _, ax = plt.subplots(figsize=(5, 5))
    set_ax(ax)

    positive = (y == 1)
    negative = (y == 0)
    X_1 = x[positive]
    X_0 = x[negative]

    plt.scatter(X_1[:, 0], X_1[:, 1], label='正例', c='c', marker='o')
    plt.scatter(X_0[:, 0], X_0[:, 1], label='反例', c='lightcoral', marker='o')

    # 绘制投影线
    x_crood = np.linspace(0, 0.15)
    # 直线经过tensor w
    y_crood = (w[0, 1] / w[0, 0]) * x_crood
    plt.plot(x_crood, y_crood, c='orange')

    # 求单位向量
    unit_w = w / np.linalg.norm(w)

    # 绘制投影线(先求对称阵)
    X_1_projecting = np.dot(X_1, np.dot(unit_w.T, unit_w))
    plt.scatter(X_1_projecting[:, 0], X_1_projecting[:, 1], c='c')
    for i in range(X_1.shape[0]):
        plt.plot([X_1[i, 0], X_1_projecting[i, 0]], [X_1[i, 1], X_1_projecting[i, 1]], c='c', linestyle='--')
    X_0_projecting = np.dot(X_0, np.dot(unit_w.T, unit_w))
    plt.scatter(X_0_projecting[:, 0], X_0_projecting[:, 1], c='lightcoral')
    for i in range(X_0.shape[0]):
        plt.plot([X_0[i, 0], X_0_projecting[i, 0]], [X_0[i, 1], X_0_projecting[i, 1]], c='lightcoral', linestyle='--')

    # 样本中心
    mu_1_center = np.dot(mu_1, np.dot(unit_w.T, unit_w))
    plt.scatter(mu_1_center[:, 0], mu_1_center[:, 1], label='正例样本中心', s=60, c='mediumblue', marker='h')
    mu_0_center = np.dot(mu_0, np.dot(unit_w.T, unit_w))
    plt.scatter(mu_0_center[:, 0], mu_0_center[:, 1], label='反例样本中心', s=60, c='red', marker='h')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([0, 1, 0, 1])

    plt.show()