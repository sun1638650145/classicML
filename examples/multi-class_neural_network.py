"""
这个例子将展示如何使用BP神经网络构建多分类的神经网络.
"""
import sys
import classicML as cml


DATASET_PATH = './datasets/iris_dataset.csv'
CALLBACKS = [cml.callbacks.History(loss_name='categorical_crossentropy',
                                   metric_name='accuracy')]

# 读取数据
ds = cml.data.Dataset(label_mode='one-hot',
                      standardization=True,
                      name='iris')
ds.from_csv(DATASET_PATH)
# 生成神经网络
model = cml.models.BPNN(seed=2021)
model.compile(network_structure=[4, 2, 3],
              optimizer='sgd',
              loss='categorical_crossentropy',
              metric='accuracy')
# 训练神经网络
model.fit(ds.x, ds.y, epochs=1000, verbose=True, callbacks=CALLBACKS)
# 可视化模型(如果您使用的是macOS或Windows, 请注释掉此句, 这句是为了在CI上测试禁用绘图提高测试速度.)
if sys.platform == 'linux':
    cml.plots.plot_history(CALLBACKS[0])
