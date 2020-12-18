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

* attribute_name: 字符串列表, 属性的名称.

### compile

```python
compile(smoothing=True, m=0)
```

编译平均独依赖估计器.

#### 参数

* smoothing: 布尔值, 是否使用平滑, 这里的实现是拉普拉斯修正.
* m: 整数, 阈值常数, 样本小于此值的属性将不会被作为超父类.

### fit

```python
fit(x, y)
```

训练平均独依赖估计器.

#### 参数

* x: 一个 Numpy数组，或者是Pandas的DataFrame, 特征数据.
* y: 一个 Numpy数组，或者是Pandas的DataFrame, 标签.

#### 返回

一个```AverageOneDependentEstimator```实例.

### predict

```python
predict(x)
```

使用平均独依赖估计器进行预测.

#### 参数

* x: 一个 Numpy数组，或者是Pandas的DataFrame, 特征数据.

#### 返回

预测的Numpy数组.

#### 异常

* ValueError: 模型没有训练的错误.