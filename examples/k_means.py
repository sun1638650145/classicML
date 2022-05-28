"""
这个例子将展示如何使用K-均值聚类.
"""
import pandas as pd

import classicML as cml

DATASET_PATH = './datasets/西瓜数据集gamma.csv'

# 读取数据.
ds = pd.read_csv(DATASET_PATH, index_col=0)
# 生成模型.
model = cml.models.KMeans(n_clusters=3)
model.compile(init=[5, 11, 23])
# 训练模型.
model.fit(ds)
