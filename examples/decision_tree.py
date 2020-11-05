"""这个例子将展示如何使用决策树进行分类."""
import numpy as np
import pandas as pd
import classicML as cml


DATASET_PATH = '../datasets/西瓜数据集.csv'
ATTRIBUTE_NAME = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感', '密度', '含糖率']

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
train_index = np.asarray([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1
validation_index = np.asarray([4, 5, 8, 9, 11, 12, 13]) - 1
x_train = dataframe.iloc[train_index, : 8]
y_train = dataframe.iloc[train_index, 8]
x_validation = dataframe.iloc[validation_index, :8]
y_validation = dataframe.iloc[validation_index, 8]
# 生成模型
model = cml.models.DecisionTreeClassifier(attribute_name=ATTRIBUTE_NAME)
model.compile(criterion='gain',
              pruning='pre')
# 训练模型
model.fit(x_train, y_train, x_validation, y_validation)
# 可视化模型
cml.plots.plot_tree(model)