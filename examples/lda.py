"""这个例子将展示如何使用线性判别分析"""
import sys
import pandas as pd
import classicML as cml


DATASET_PATH = '../datasets/西瓜数据集alpha.csv'

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :2].values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成模型
model = cml.models.LDA()
# 训练模型
model.fit(x, y)
# 可视化模型(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_lda(model, x, y, '密度', '含糖率')