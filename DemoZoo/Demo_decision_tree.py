import pandas as pd
import classicML as cml

DATASET_PATH = '西瓜数据集.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0)
row = df.shape[1]
x = df.iloc[:, :row-1].values
y = df.iloc[:, row-1].values
# 生成决策树
tree = cml.DecisionTree()
tree.compile(critertion='gini',
             pruning=None,
             feature_attr=['脐部', '色泽', '根蒂', '敲声', '纹理', '触感'])
# 训练
tree.fit(x, y)
# 绘图
cml.plot_decision_tree(tree.tree)
# 测试
x_test = ['稍凹', '浅白', '稍蜷', '浊响', '清晰', '软粘']
ans = tree.predict(x_test)
print(ans)