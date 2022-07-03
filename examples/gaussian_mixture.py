"""
这个例子将展示如何使用高斯混合聚类.
"""
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
print(model.clusters)
