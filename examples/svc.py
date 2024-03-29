"""
这个例子将展示如何使用支持向量分类器.
"""
import sys

import classicML as cml


DATASET_PATH = './datasets/西瓜数据集alpha.csv'

# 读取数据
ds = cml.data.Dataset(label_mode='max-margin')
ds.from_csv(DATASET_PATH)
# 生成模型
model = cml.models.SVC(seed=2021)
model.compile(C=10000.0, kernel='rbf')
# 训练模型
model.fit(ds.x, ds.y)
# 可视化模型(如果您使用的是macOS或Windows, 请注释掉此句, 这句是为了在CI上测试禁用绘图提高测试速度.)
if sys.platform == 'linux':
    cml.plots.plot_svc(model, ds.x, ds.y, '密度', '含糖率')
