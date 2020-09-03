import pandas as pd
import classicML as cml


DATASET_PATH = '西瓜数据集a.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0)
x = df.iloc[:, :2].values
y = df.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成逻辑回归模型
model = cml.LogisticRegression()
model.compile(optimizer='Newton',
              learning_rate=1e-2)
# 训练
model.fit(x, y, epochs=10000, verbose=True)
# 绘图
cml.plot_logistic_regression(model.beta, x, y)