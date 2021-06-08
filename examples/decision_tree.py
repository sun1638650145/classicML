"""
这个例子将展示如何使用决策树进行分类.
"""
import sys
import numpy as np
import pandas as pd
import classicML as cml


DATASET_PATH = './datasets/西瓜数据集.tsv'
ATTRIBUTE_NAME = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感', '密度', '含糖率']

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, sep='\t', index_col=0, header=0)
train_index = np.asarray([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1
validation_index = np.asarray([4, 5, 8, 9, 11, 12, 13]) - 1

train_ds = cml.data.Dataset(dataset_type='train')
val_ds = cml.data.Dataset(dataset_type='val')
train_ds.from_dataframe(dataframe.iloc[train_index])
val_ds.from_dataframe(dataframe.iloc[validation_index])
# 生成模型
model = cml.models.DecisionTreeClassifier(attribute_name=ATTRIBUTE_NAME)
model.compile(criterion='gain',
              pruning='pre')
# 训练模型
model.fit(train_ds.x, train_ds.y, val_ds.x, val_ds.y)
# 可视化模型(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_tree(model)
