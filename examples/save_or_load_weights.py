"""这个例子将展示如何保存(加载)权重."""
import pandas as pd
import classicML as cml

DATASET_PATH = './datasets/西瓜数据集alpha.csv'

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :2].values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成模型
model = cml.models.LogisticRegression(seed=1234)
model.compile(optimizer='newton',
              loss='log_likelihood',
              metric='accuracy')
# 训练模型
model.fit(x, y, epochs=10, verbose=True, callbacks=None)
# 保存权重(当你需要加载的时候, 创建模型后使用load_weights方法加载模型)
model.save_weights('./weights.h5')