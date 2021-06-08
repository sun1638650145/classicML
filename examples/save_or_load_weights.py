"""
这个例子将展示如何保存(加载)权重.
"""
import tempfile

import classicML as cml


DATASET_PATH = './datasets/西瓜数据集alpha.csv'

# 读取数据
ds = cml.data.Dataset()
ds.from_csv(DATASET_PATH)
# 生成模型
model = cml.models.LogisticRegression(seed=2021)
model.compile(optimizer='newton',
              loss='log_likelihood',
              metric='accuracy')
# 训练模型
model.fit(ds.x, ds.y, epochs=10, verbose=True, callbacks=None)
# 保存权重(当你需要加载的时候, 创建模型后使用load_weights方法加载模型)
model.save_weights(tempfile.mkstemp()[1])
