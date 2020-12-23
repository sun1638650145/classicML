"""这个例子将展示如何使用BP神经网络构建多分类的神经网络"""
import sys
import numpy as np
import pandas as pd
import classicML as cml


DATASET_PATH = './datasets/iris_dataset.csv'
CALLBACKS = [cml.callbacks.History(loss_name='categorical_crossentropy',
                                   metric_name='accuracy')]

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :-1]
x = (x - np.mean(x, axis=0)) / np.var(x, axis=0)
y = dataframe.iloc[:, -1]
y = pd.get_dummies(y)
# 生成神经网络
model = cml.BPNN(seed=2020)
model.compile(network_structure=[4, 2, 3],
              optimizer='sgd',
              loss='categorical_crossentropy',
              metric='accuracy')
# 训练神经网络
model.fit(x.values, y.values, epochs=1000, verbose=True, callbacks=CALLBACKS)
# 可视化历史记录(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_history(CALLBACKS[0])