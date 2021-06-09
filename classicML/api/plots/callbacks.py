from classicML.api.plots import _plt as plt
from classicML.api.plots.utils import _set_history_axis_and_background
from classicML.api.plots.utils import _history_plot_config


def plot_history(history):
    """可视化历史记录.

    Arguments:
        history: classicML.backend.callbacks.History 实例.
    """
    _, ax = plt.subplots()
    _set_history_axis_and_background(ax)

    # 绘制损失曲线
    ax.plot(history.loss, label=history.loss_name, c='lightcoral')
    # 绘制评估曲线
    ax.plot(history.metric, label=history.metric_name, c='c')

    _history_plot_config()
    plt.show()
