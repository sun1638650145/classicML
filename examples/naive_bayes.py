"""这个例子将展示如何使用朴素贝叶斯分类器."""
import pandas as pd
import classicML as cml

DATASET_PATH = '../datasets/西瓜数据集alpha.csv'
ATTRIBUTE_NAME = ['密度', '含糖率']

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成模型
model = cml.NB(attribute_name=ATTRIBUTE_NAME)
model.compile()
# 训练模型
model.fit(x, y)
# 可视化模型
cml.plots.plot_bayes(model, x, y)