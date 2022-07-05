"""
这个例子将展示如何使用高斯混合聚类.
"""
import sys

import classicML as cml

DATASET_PATH = './datasets/西瓜数据集gamma.csv'

# 读取数据.
ds = cml.data.Dataset(label_mode='unsupervised')
ds.from_csv(DATASET_PATH)
# 生成模型.
model = cml.models.GaussianMixture()
model.compile(init=[5, 21, 26])
# 训练模型.
model.fit(ds.x)
# 可视化模型(如果您使用的是macOS或Windows, 请注释掉此句, 这句是为了在CI上测试禁用绘图提高测试速度.)
if sys.platform == 'linux':
    cml.plots.plot_gaussian_mixture(model, ds.x, '密度', '含糖率')
