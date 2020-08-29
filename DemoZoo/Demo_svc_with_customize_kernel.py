import numpy as np
import pandas as pd
import classicML as cml


def my_kernel(x_i, x_j):
    """
        自定义核函数
        linear + poly
    """
    return np.dot(x_j, x_i.T) + np.power(np.dot(x_j, x_i.T), 3)


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
              kernel='customize',
              customize_kernel=my_kernel)
# 训练
model.fit(x, y)
# 绘图
cml.plot_svc(model, x, y)