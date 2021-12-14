"""
这个例子将展示如何使用BP神经网络构建简单的感知机.
"""
import sys

import classicML as cml


DATASET_PATH = './datasets/西瓜数据集.tsv'
CALLBACKS = [cml.callbacks.History(loss_name='crossentropy',
                                   metric_name='accuracy')]

# 读取数据
ds = cml.data.Dataset(digitization=True)
ds.from_csv(DATASET_PATH, sep='\t')
# 生成神经网络
model = cml.models.BPNN(seed=2021, initializer='glorot_normal')
model.compile(network_structure=[3, 1],
              optimizer='adam',
              loss='crossentropy',
              metric='accuracy')
# 训练神经网络
model.fit(ds.x, ds.y, epochs=2500, verbose=True, callbacks=CALLBACKS)
# 可视化模型(如果您使用的是macOS或Windows, 请注释掉此句, 这句是为了在CI上测试禁用绘图提高测试速度.)
if sys.platform == 'linux':
    cml.plots.plot_history(CALLBACKS[0])
