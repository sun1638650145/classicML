# models

在 classicML 中最基本也是最重要的就是模型，通过预定义的模型你可以实现你的机器学习任务. 你可以直接实例化一个你想使用的模型，比如：

```python
import classicML as cml

# 实例化一个神经网络
model = cml.models.BPNN()
# 实例化一个支持向量机
model = cml.models.SVC()
# 实例化一个决策树
model = cml.models.DecisionTreeClassifier()
```

目前，在 classicML 中大部分模型都有六个类方法```model.compile()```，```model.fit()```，```model.predict()```，```model.score()```，```model.save_weights()```，```model.load_weights()```；前四个类方法分别控制模型工作流程中的编译模型参数，训练模型，使用训练好的模型进行预测推理和在预测模式下计算准确率；后两个方法可以实现模型权重的保存和加载. 以支持向量机为例，流程大概是这样的：

```python
import classicML as cml
# 实例化一个支持向量机
model = cml.models.SVC(seed=2020)
# 编译模型参数，配置软间隔系数和核函数
model.compile(C=10000.0, kernel='rbf')
# 训练模型
model.fit(x, y)
# 在测试集上测试
y_pred = model.predict(x_test)
# 保存模型权重
model.save_weights('./weights.h5')
```

注意, `DecisionStumpClassifier`和`TwoLevelDecisionTreeClassifier`被设计用来作为基学习器使用, 不推荐直接使用.

## BaseModel

classicML的模型抽象基类, classicML的模型全部继承于此. 通过继承`BaseModel`, 至少实现`fit`和`predict`方法就可以构建自己的模型.

```python
cml.models.BaseModel()
```

### compile

```python
compile(**kwargs)
```

编译模型.

### fit

```python
fit(x, y, **kwargs)
```

训练模型.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 异常

* <b>NotImplementedError</b>: 需要用户自行实现.

### predict

```python
predict(x)
```

使用模型进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* <b>NotImplementedError</b>: 需要用户自行实现.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>NotImplemented</b>: 需要用户自行实现.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存. 如果您希望您的模型参数收到保护, 可自行实现模型的保存方式; 如果您希望您的模型开源, 请参照`cml.backend.io`的协议方式实现参数的保存.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>NotImplemented</b>: 需要用户自行实现.

## AdaBoostClassifier

`AdaBoost`分类器.

```python
cml.models.AdaBoostClassifier()
```

### compile

```python
compile(base_algorithm=TwoLevelDecisionTreeClassifier)
```

编译`AdaBoost`分类器.

#### 参数

* <b>base_algorithm</b>: `BaseLearner`对象, `AdaBoost`使用的基学习器算法; 目前只实现了`TwoLevelDecisionTreeClassifier`, 未来将接入更多算法.

### fit

```python
fit(x, y, max_estimators=50)
```

训练`AdaBoost`分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.
* <b>y</b>: 一个 Numpy数组, 标签.
* <b>max_estimators</b>: 整数, 基学习器集成的最大数量(训练的最大轮数).

#### 返回

一个`AdaBoostClassifier`实例.

### predict

```python
predict(x)
```

使用`AdaBoost`分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.

#### 返回

`AdaBoostClassifier`预测的结果.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

## AveragedOneDependentEstimator

平均独依赖估计器，一种半朴素贝叶斯分类器.

```python
cml.models.AveragedOneDependentEstimator(attribute_name=None)  # 可以使用缩写 cml.models.AODE()
```

### 参数

* <b>attribute_name</b>: 字符串列表，属性的名称.

### compile

```python
compile(smoothing=True, m=0)
```

编译平均独依赖估计器.

#### 参数

* <b>smoothing</b>: 布尔值，是否使用平滑，这里的实现是拉普拉斯修正.
* <b>m</b>: 整数，阈值常数，样本小于此值的属性将不会被作为超父类.

### fit

```python
fit(x, y)
```

训练平均独依赖估计器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 返回

一个```AverageOneDependentEstimator```实例.

### predict

```python
predict(x)
```

使用平均独依赖估计器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## BackPropagationNeuralNetwork

BP神经网络.

```python
cml.models.BackPropagationNeuralNetwork(seed=None, initializer=None)  # 可以使用缩写 cml.models.BPNN()
```

### 参数

* <b>seed</b>: 整数，随机种子.
* <b>initializer</b>: 字符串，或者```cml.initializers.Initializer```实例，初始化器.

### compile

```python
compile(network_structure, optimizer='sgd', loss='crossentropy', metric='accuracy')
```

编译神经网络，配置训练时使用的超参数.

#### 参数

* <b>network_structure</b>: 列表，神经网络的结构，定义神经网络的隐含层和输出层的神经元个数(输入层目前将自动推理)，

  例如：

  ​	[3, 1]是一个隐含层3个神经元和输出层1个神经元的网络，

  ​	[5, 5, 2]是一个有两个隐含层每层有5个神经元和输出层2个神经元的网络.

* <b>optimizer</b>: 字符串，或者```cml.optimizers.Optimizer```实例，神经网络使用的优化器.

* <b>loss</b>: 字符串，或者```cml.losses.Loss```实例，神经网络使用的损失函数.

* <b>metric</b>: 字符串，或者```cml.metrics.Metric```实例，神经网络使用的评估函数.

### fit

```python
fit(x, y, epochs=1, verbose=True, callbacks=None)
```

训练神经网络.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>epochs</b>: 整数，训练的轮数.
* <b>verbose</b>: 布尔值（可选参数），显示日志信息.
* <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

一个```BackPropagationNeuralNetwork```实例.

### predict

```python
predict(x)
```

使用神经网络进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组（以概率形式）.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.
* <b>TypeError</b>: 输入参数的类型错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## BaggingClassifier

`Bagging`分类器.

```python
cml.models.BaggingClassifier()
```

### compile

```python
compile(self, base_algorithm=DecisionStumpClassifier, seed=np.random.randint(65535))
```

编译`Bagging`分类器.

#### 参数

* <b>base_algorithm</b>: `BaseLearner`对象, `Bagging`使用的基学习器算法; 目前只实现了`DecisionStumpClassifier`, 未来将接入更多算法.

### fit

```python
fit(x, y, n_estimators=10)
```

训练`Bagging`分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.
* <b>y</b>: 一个 Numpy数组, 标签.
* <b>n_estimators</b>: 整数, 基学习器集成的数量.

#### 返回

一个`BaggingClassifier`实例.

### predict

```python
predict(x)
```

使用`Bagging`分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.

#### 返回

`BaggingClassifier`预测的结果.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

## DecisionStumpClassifier

决策树桩分类器.

```python
cml.models.DecisionStumpClassifier()
```

### fit

```python
fit(x, y)
```

训练决策树桩分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 返回

一个```DecisionStumpClassifier```实例.

#### 异常

* <b>AttributeError</b>: 没有验证集.

### predict

```python
predict(x)
```

使用决策树桩分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

`DecisionStumpClassifier`预测的结果.

## DecisionTreeClassifier

决策树分类器.

```python
cml.models.DecisionTreeClassifier(attribute_name=None)
```

### 参数

* <b>attribute_name</b>: 字符串列表，属性的名称.

### compile

```python
compile(criterion='gain', pruning=None)
```

编译决策树, 配置训练时使用的超参数.

#### 参数

* <b>criterion</b>: {'gain', 'gini', 'entropy'}，决策树学习的划分方式.
* <b>pruning</b>: {None, 'pre', 'post'}，是否对决策树进行剪枝操作，None表示不使用剪枝.

#### 异常

* <b>AttributeError</b>: 参数错误.

### fit

```python
fit(x, y, x_validation=None, y_validation=None)
```

训练决策树分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.
* <b>x_validation</b>: 一个 Numpy数组，或者是Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: 一个 Numpy数组，或者是Pandas的DataFrame，剪枝使用的验证标签.

#### 返回

一个```DecisionTreeClassifier```实例.

#### 异常

* <b>AttributeError</b>: 没有验证集.

### predict

```python
predict(x)
```

使用分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 参考文献

* [如何存储原始的二进制数据](https://docs.h5py.org/en/2.3/strings.html)

#### 注意

* 模型将不会保存关于优化器的超参数.

## KMeans

K-均值聚类.

```python
cml.models.KMeans(n_clusters=3)
```

### 参数

* <b>n_clusters</b>: 整数, 聚类簇的数量.

### compile

```python
compile(init='random', tol=1e-3)
```

编译K-均值聚类.

#### 参数

* <b>criterion</b>: {'gain', 'gini', 'entropy'}，决策树学习的划分方式.
* <b>pruning</b>: {None, 'pre', 'post'}，是否对决策树进行剪枝操作，None表示不使用剪枝.
* <b>init</b>: 均值向量的初始化方式
    * 'random': 随机初始化;
    * 列表或一个 Numpy数组: 可以指定训练数据的索引或者值, 也可以直接给定具体的均值向量.

* <b>tol</b>: 浮点数, 停止训练的最小调整幅度阈值.

### fit

```python
fit(x, epochs=100)
```

训练K-均值聚类.

#### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.
* <b>epochs</b>: 整数, 最大的训练轮数, 如果均值向量已经不更新将会提前自动结束训练.

#### 返回

一个```K-means```实例.

### predict

```python
predict(x)
```

使用K-均值聚类预测新样本所在的簇.

#### 参数

* <b>x</b>: 一个 Numpy数组或者列表，特征数据.

#### 返回

新样本所在的簇.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

## LinearDiscriminantAnalysis

线性判别分析.

```python
cml.models.LinearDiscriminantAnalysis()  # 可以使用缩写 cml.models.LDA()
```

### fit

```python
fit(x, y)
```

训练模型.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 返回

一个```LinearDiscriminantAnalysis```实例.

### predict

```python
predict(x)
```

模型进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

## LogisticRegression

逻辑回归.

```python
cml.models.LogisticRegression(seed=None, initializer=None)
```

### 参数

* <b>seed</b>: 整数，随机种子.
* <b>initializer</b>: 字符串，或者```cml.initializers.Initializer```实例，初始化器.

### compile

```python
compile(optimizer='newton', loss='log_likelihood', metric='accuracy')
```

编译模型，配置训练时使用的超参数.

#### 参数

* <b>optimizer</b>: 字符串，或者```cml.optimizers.Optimizer```实例，模型使用的优化器.

* <b>loss</b>: 字符串，或者```cml.losses.Loss```实例，模型使用的损失函数.

* <b>metric</b>: 字符串，或者```cml.metrics.Metric```实例，模型使用的评估函数.

### fit

```python
fit(x, y, epochs=1, verbose=True, callbacks=None)
```

训练神经网络.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>epochs</b>: 整数，训练的轮数.
* <b>verbose</b>: 布尔值（可选参数），显示日志信息.
* <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

一个```LogisticRegression```实例.

### predict

```python
predict(x)
```

使用神经网络进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组（以概率形式）.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## NaiveBayesClassifier

朴素贝叶斯分类器.

```python
cml.models.NaiveBayesClassifier(attribute_name=None)  # 可以使用缩写 cml.models.NB()
```

### 参数

* <b>attribute_name</b>: 字符串列表，属性的名称.

### compile

```python
compile(smoothing=True)
```

编译朴素贝叶斯分类器.

#### 参数

* <b>smoothing</b>: 布尔值，是否使用平滑，这里的实现是拉普拉斯修正.

### fit

```python
fit(x, y)
```

训练朴素贝叶斯分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 返回

一个```NaiveBayesClassifier```实例.

### predict

```python
predict(x, probability=False)
```

使用朴素贝叶斯分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>probability</b>: 布尔值，是否使用归一化的概率形式.

#### 返回

预测的Numpy数组，不使用概率形式将返回0或1的标签数组, 使用将返回反正例概率的数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## RadialBasisFunctionNetwork

径向基函数网络.

```python
cml.models.RadialBasisFunctionNetwork(seed=None)  # 可以使用缩写 cml.models.RBF()
```

### 参数

* <b>seed</b>: 整数，随机种子.

### compile

```python
compile(hidden_units, optimizer='rbf', loss='mse', metric='accuracy')
```

编译径向基函数网络, 配置训练时使用的超参数.

#### 参数

* <b>hidden_units</b>: 整数，径向基函数网络的隐含层神经元数量.

* <b>optimizer</b>: 字符串，或者```cml.optimizers.Optimizer```实例，径向基函数网络使用的优化器.

* <b>loss</b>: 字符串，或者```cml.losses.Loss```实例，径向基函数网络使用的损失函数.

* <b>metric</b>: 字符串，或者```cml.metrics.Metric```实例，径向基函数网络使用的评估函数.

#### 注意

* 注意RBF只能使用```cml.optimizers.RadialBasisFunctionOptimizer```优化器，之所以开放优化器选项，只是为了满足用户修改学习率的需求.
* 使用交叉熵作为损失函数有潜在异常的可能性，除隐含层神经元个数和学习率之外，建议使用默认参数.

### fit

```python
fit(x, y, epochs=1, verbose=True, callbacks=None)
```

训练径向基函数网络.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>epochs</b>: 整数，训练的轮数.
* <b>verbose</b>: 布尔值（可选参数），显示日志信息.
* <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

一个```RadialBasisFunctionNetwork```实例.

### predict

```python
predict(x)
```

使用径向基函数网络进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组（以概率形式）.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## SuperParentOneDependentEstimator

超父独依赖估计器，一种半朴素贝叶斯分类器.

```python
cml.models.SuperParentOneDependentEstimator(attribute_name=None)  # 可以使用缩写 cml.models.SPODE()
```

### 参数

* <b>attribute_name</b>: 字符串列表，属性的名称.

### compile

```python
compile(super_parent_name, smoothing=True)
```

编译超父独依赖估计器.

#### 参数

* <b>super_parent_name</b>: 字符串，超父的名称.
* <b>smoothing</b>: 布尔值，是否使用平滑，这里的实现是拉普拉斯修正.

### fit

```python
fit(x, y)
```

训练超父独依赖估计器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 返回

一个```SuperParentOneDependentEstimator```实例.

### predict

```python
predict(x, probability=False)
```

使用超父独依赖估计器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>probability</b>: 布尔值，是否使用归一化的概率形式.

#### 返回

预测的Numpy数组，不使用概率形式将返回0或1的标签数组, 使用将返回反正例概率的数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## SupportVectorClassifier

支持向量分类器.

```python
cml.models.SupportVectorClassifier(seed=None)  # 可以使用缩写 cml.models.SVC()
```

### 参数

* <b>seed</b>: 整数，随机种子.

### compile

```python
compile(C=1.0, kernel='rbf', gamma='auto', tol=1e-3)
```

编译分类器, 配置训练时使用的超参数.

#### 参数

* <b>C</b>: 浮点数，软间隔正则化系数.
* <b>kernel</b>: 字符串，或者```cml.kernel.Kernels```实例，分类器使用的核函数.
* <b>gamma</b>: {'auto', 'scale'} 或者浮点数，在使用高斯(径向基)核, sigmoid核或者多项式核时,的核函数系数，使用其他核函数时无效.
* <b>tol</b>: 浮点数，停止训练的最大误差值.

### fit

```python
fit(x, y, epochs=1000)
```

训练分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

* <b>y</b>: 一个 Numpy数组，标签.

* <b>epochs</b>: 整数，最大的训练轮数，如果是-1则表示需要所有的样本满足条件时，才能停止训练，即没有限制.

#### 返回

一个```SupportVectorClassifier```实例.

### predict

```python
predict(x)
```

使用分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### score

```python
score(x, y)
```

在预测模式下计算准确率.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.

#### 返回

当前的准确率.

### load_weights

```python
load_weights(filepath)
```

加载模型参数.

#### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.

#### 异常

* <b>KeyError</b>: 模型权重加载失败.

#### 注意

* 模型将不会加载关于优化器的超参数.

### save_weights

```python
save_weights(filepath)
```

将模型权重保存为一个HDF5文件.

#### 参数

* <b>filepath</b>: 字符串，权重文件保存的路径.

#### 异常

* <b>KeyError</b>: 模型权重保存失败.

#### 注意

* 模型将不会保存关于优化器的超参数.

## TwoLevelDecisionTreeClassifier

2层决策树分类器.

```python
cml.models.TwoLevelDecisionTreeClassifier()
```

### fit

```python
fit(x, y, **kwargs)
```

训练2层决策树分类器.

#### 参数

* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.
* <b>sample_distribution</b>: 一个 Numpy数组，样本分布.

#### 返回

一个```TwoLevelDecisionTreeClassifier```实例.

### predict

```python
predict(x)
```

使用决策树桩分类器进行预测.

#### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.

#### 返回

`TwoLevelDecisionTreeClassifier`预测的结果.