"""
这个例子将展示如何使用径向基函数网络拟合一个异或数据集.
"""
import sys
import classicML as cml


CALLBACKS = [cml.callbacks.History(loss_name='mse',
                                   metric_name='accuracy')]

# 读取数据
ds = cml.data.Dataset()
ds.from_tensor_slices(x=[[0, 0], [0, 1], [1, 0], [1, 1]],
                      y=[0, 1, 1, 0])
# 生成神经网络
model = cml.models.RBF(seed=2021)
model.compile(hidden_units=16)
# 训练神经网络
model.fit(ds.x, ds.y, epochs=1000, verbose=True, callbacks=CALLBACKS)
# 可视化历史记录(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_history(CALLBACKS[0])
