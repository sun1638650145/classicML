"""这个例子将展示如何使用自定义核函数的支持向量分类器."""
import sys
import numpy as np
import pandas as pd
import classicML as cml


DATASET_PATH = './datasets/西瓜数据集alpha.csv'


# 自定义核函数
class MyKernel(cml.kernels.Kernel):
    def __call__(self, x_i, x_j):
        """只需要实现__call__方法即可."""
        kappa = np.matmul(x_j, x_i.T) + np.power(np.matmul(x_j, x_i.T), 3)

        return kappa


# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :2].values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = -1
# 生成模型
model = cml.models.SVC(seed=2020)
model.compile(C=10000.0, kernel=MyKernel('my_kernel'))
# 训练模型
model.fit(x, y, epochs=-1)
# 可视化模型(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_svc(model, x, y, '密度', '含糖率')