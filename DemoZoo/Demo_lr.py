import pandas as pd
from lm_model.logistic_regression import LogisticRegression


DATASET_PATH = '西瓜数据集a.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0)
x = df.iloc[:, :2].values
y = df.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成逻辑回归模型
model = LogisticRegression(seed=16)
model.compile(optimizer='GD',
              learning_rate=1e-2)
# 训练
model.fit(x, y, epochs=10000, verbose=True)
