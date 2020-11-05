"""这个例子将展示如何使用径向基函数网络拟合一个异或数据集"""
import classicML as cml


CALLBACKS = [cml.callbacks.History(loss_name='mse',
                                   metric_name='accuracy')]

# 读取数据
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]
# 生成神经网络
model = cml.models.RBF(seed=2020)
model.compile(hidden_units=16)
# 训练神经网络
model.fit(x, y, epochs=1000, verbose=True, callbacks=CALLBACKS)
# 可视化历史记录
cml.plots.plot_history(CALLBACKS[0])