"""
这个例子将展示如何使用AdaBoost分类器.
"""
import classicML as cml

DATASET_PATH = './datasets/西瓜数据集alpha.csv'

# 读取数据.
ds = cml.data.Dataset(label_mode='max-margin')
ds.from_csv(DATASET_PATH)
# 生成模型.
model = cml.AdaBoostClassifier()
model.compile()
# 训练模型.
model.fit(ds.x, ds.y)
print(model.alpha_list)
