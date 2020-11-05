"""这个例子将展示如何使用支持向量分类器."""
import pandas as pd
import classicML as cml


DATASET_PATH = '../datasets/西瓜数据集alpha.csv'

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :2].values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = -1
# 生成模型
model = cml.models.SVC(seed=2020)
model.compile(C=10000.0, kernel='rbf')
# 训练模型
model.fit(x, y)
# 可视化模型
cml.plots.plot_svc(model, x, y, '密度', '含糖率')