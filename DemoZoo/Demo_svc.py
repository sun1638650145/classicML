import pandas as pd
import classicML as cml


DATASET_PATH = '西瓜数据集a.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = df.iloc[:, :2].values
y = df.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = -1
# 生成支持向量分类器
model = cml.SVC(seed=16)
model.compile(C=10000.0,
              kernel='rbf')
# 训练
model.fit(x, y)
# 绘图
cml.plot_svc(model, x, y)