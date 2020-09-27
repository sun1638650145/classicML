import pandas as pd
import classicML as cml


DATASET_PATH = '西瓜数据集a.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = df.iloc[:, :2].values
y = df.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成线性判别分析模型
model = cml.LDA()
# 训练
model.fit(x, y)
# 绘图
cml.plot_linear_discriminant_analysis(model, x, y, '密度', '含糖率')