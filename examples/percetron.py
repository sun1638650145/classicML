"""这个例子将展示如何使用BP神经网络构建简单的感知机"""
import sys

import pandas as pd
import classicML as cml


DATASET_PATH = './datasets/西瓜数据集.csv'
CALLBACKS = [cml.callbacks.History(loss_name='crossentropy',
                                   metric_name='accuracy')]

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
dataframe = pd.get_dummies(dataframe, columns=['色泽', '根蒂', '敲声', '纹理', '触感', '脐部'])
x = dataframe.drop('好瓜', axis=1).values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成神经网络
model = cml.models.BPNN(seed=2020, initializer='glorot_normal')
model.compile(network_structure=[3, 1],
              optimizer='adam',
              loss='crossentropy',
              metric='accuracy')
# 训练神经网络
model.fit(x, y, epochs=2500, verbose=True, callbacks=CALLBACKS)
# 可视化历史记录(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_history(CALLBACKS[0])