"""
这个例子将展示如何使用AODE分类器.
"""
import classicML as cml

DATASET_PATH = './datasets/西瓜数据集.tsv'
ATTRIBUTE_NAME = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感', '密度', '含糖率']

# 读取数据
ds = cml.data.Dataset()
ds.from_csv(DATASET_PATH, sep='\t')
# 生成模型
model = cml.models.AODE(attribute_name=ATTRIBUTE_NAME)
model.compile(smoothing=True)
# 训练模型
model.fit(ds.x, ds.y)
