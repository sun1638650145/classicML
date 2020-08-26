import numpy as np
import matplotlib.pyplot as plt
# 设置中文
plt.rcParams['font.family'] = 'Arial Unicode MS'


def set_ax(ax):
    # 隐藏坐标轴
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')

    ax.patch.set_facecolor('gray')
    ax.patch.set_alpha(0.1)
    ax.grid(axis='y', linestyle='dotted')


def plot_support_vector_classification(svc, x, y):
    """可视化支持向量机"""
    x = np.asarray(x)
    if x.shape[1] != 2:
        raise Exception('目前暂时不能可视化非二维数据')

    kernel = svc.kernel
    C = svc.C
    ax = plt.subplot()
    set_ax(ax)

    positive = (y == 1)
    negative = (y == -1)

    # 生成向量空间
    x_crood = np.linspace(start=0, stop=1, num=300)  # x0
    y_crood = np.linspace(start=0, stop=0.8, num=300)  # x1
    vector_matrix = np.meshgrid(x_crood, y_crood)

    # 进行预测并改变shape
    y_pred = svc.predict(np.c_[vector_matrix[0].ravel(), vector_matrix[1].ravel()])
    z = y_pred.reshape(vector_matrix[0].shape)

    # 使用向量空间的这些点进行预测的结果绘制决策边界(z是到平面高度，代替直接绘制y=0)
    CS = ax.contour(vector_matrix[0], vector_matrix[1], z, [0], colors='orange', linewidths=1)
    ax.clabel(CS, fmt={CS.levels[0]: '决策边界-decision boundary'})

    # 添加数据和支持向量
    ax.scatter(x[positive, 0], x[positive, 1], label='正例', color='c')
    ax.scatter(x[negative, 0], x[negative, 1], label='反例', color='lightcoral')
    ax.scatter(x[svc.support_, 0], x[svc.support_, 1], label='支持向量', marker='o', color='', edgecolors='green', s=150)

    ax.legend()
    ax.set_title('kernel={}, C={}'.format(kernel, C))

    plt.show()


# alias
plot_svc = plot_support_vector_classification