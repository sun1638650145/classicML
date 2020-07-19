import pandas as pd
import classicML as cml

DATASET_PATH = '西瓜数据集.csv'
# 读取数据
df = pd.read_csv(DATASET_PATH, index_col=0)
row = df.shape[1]
x = df.iloc[:, :row-1]
y = df.iloc[:, row-1]
# 生成决策树
tree = cml.DecisionTree(critertion='gini', pruning=None)
# 训练
tree.fit(x, y)
# 绘图
cml.plot_decision_tree(tree.tree)
# 测试
x_test_2 = ['稍凹', '浅白', '稍蜷', '浊响', '清晰', '软粘']
ans = tree.predict(x_test_2)
print(ans)