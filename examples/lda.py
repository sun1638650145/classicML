"""
这个例子将展示如何使用线性判别分析.
"""
import sys
import classicML as cml


DATASET_PATH = './datasets/西瓜数据集alpha.csv'

# 读取数据
ds = cml.data.Dataset()
ds.from_csv(DATASET_PATH)
# 生成模型
model = cml.models.LDA()
# 训练模型
model.fit(ds.x, ds.y)
# 可视化模型(如果您使用的是MacOS, 请注释掉此句, 这句是为了在CI上测试用的.)
if sys.platform != 'darwin':
    cml.plots.plot_lda(model, ds.x, ds.y, '密度', '含糖率')
