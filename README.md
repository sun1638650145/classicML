# classicML 
Python简单易用的经典机器学习框架
## 重要信息

你可以使用pip安装

```shell
pip install classicML
```

## version v0.1
1. 添加决策树
2. 决策树支持离散值、连续值
3. 基于信息熵、信息增益、基尼指数划分；支持预剪枝和后剪枝；暂不支持多变量决策树和缺失值处理（建议在读入数据集之前处理）

## version v0.2
1. 添加神经网络
2. 神经网络支持交叉熵损失函数和均方误差损失函数；支持的优化器有GradientDescent、SGD、Adam

### version v0.2.2
1. 发行版发布到PyPi

### version v0.2.3

1. 添加径向基函数神经网络
2. 例行修复BUG

### version v0.2.4

1. 重写sklearn依赖函数，添加到DecisionTree.tree_model.backend，显著减少安装后实际的环境大小

## version v0.3

### rc1

1. DecisionTree: 优化API调用方式，是语法更统一；增加决策树的输入特征数据类型，理论上现在支持一切的array-like的数据类型
2. NeuralNetwork: verbose支持显示预计时间；BPNN支持自定义损失函数

### rc2

3. SupportVectorMachine: 添加支持向量分类器
4. 支持的核函数有线性核、多项式核、高斯核