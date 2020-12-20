# models

在 classicML 中最基本也是最重要的就是模型，通过预定义的模型你可以实现你的机器学习任务。你可以直接实例化一个你想使用的模型，比如：

```python
import classicML as cml

# 实例化一个神经网络
model = cml.models.BPNN()
# 实例化一个支持向量机
model = cml.models.SVC()
# 实例化一个决策树
model = cml.models.DecisionTreeClassifier()
```

目前，在 classicML 中大部分模型都有三个类方法```model.compile()```，```model.fit()```，```model.predict()```，这三个类方法分别控制模型工作流程中的编译模型参数，训练模型，使用训练好的模型进行预测推理。以支持向量机为例，流程大概是这样的：

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
```

## AveragedOneDependentEstimator

平均独依赖估计器，一种半朴素贝叶斯分类器。

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

## BackPropagationNeuralNetwork

BP神经网络。

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

* network_structure: 列表，神经网络的结构，定义神经网络的隐含层和输出层的神经元个数(输入层目前将自动推理)；例如：

  [3, 1]是一个隐含层3个神经元和输出层1个神经元的网络，

  [5, 5, 2]是一个有两个隐含层每层有5个神经元和输出层2个神经元的网络.

* optimizer: 字符串，或者```cml.optimizers.Optimizer```实例，神经网络使用的优化器.

* loss: 字符串，或者```cml.losses.Loss```实例，神经网络使用的损失函数.

* metric: 字符串，或者```cml.metrics.Metric```实例，神经网络使用的评估函数.

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

## DecisionTreeClassifier

决策树分类器。

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

* ValueError: 模型没有训练的错误.