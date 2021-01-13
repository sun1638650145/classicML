# metrics

classicML的评估函数.

## Metric

评估函数的基类.

```python
cml.metrics.Metric(name='metric')
```

### 参数

* <b>name</b>: 字符串，评估函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## Accuracy

准确率评估函数，将根据标签的实际形状自动使用二分类或者多分类评估函数.

```python
cml.metrics.Accuracy(name='accuracy')
```

### 参数

* <b>name</b>: 字符串，评估函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的准确率.

## BinaryAccuracy

二分类准确率评估函数.

```python
cml.metrics.BinaryAccuracy(name='binary_accuracy')
```

### 参数

* <b>name</b>: 字符串，评估函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的准确率.

## CategoricalAccuracy

多分类准确率评估函数.

```
cml.metrics.CategoricalAccuracy(name='categorical_accuracy')
```

### 参数

- **name**: 字符串，评估函数名称.

### \__call__

函数实现.

```
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

- **y_pred**: 一个 Numpy数组，预测的标签.
- **y_true**: 一个 Numpy数组，真实的标签.

#### 返回

当前的准确率.