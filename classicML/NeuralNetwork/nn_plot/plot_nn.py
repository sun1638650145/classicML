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


def plot_history(losses, labels):
    ax = plt.subplot()
    for i in range(len(losses)):
        ax.plot(losses[i], label=labels[i])
    set_ax(ax)

    ax.legend(loc='best')
    ax.set_xlabel(xlabel='epochs')
    plt.show()
