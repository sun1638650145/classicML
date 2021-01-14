# training

classicML的```training```模块用于安全的智能的访问classicML的后端函数.

## get_criterion

获取使用的划分选择方式.

```python
get_criterion(criterion)
```

### 参数

* <b>criterion</b>: 字符串，决策树学习的划分方式.

### 异常

* <b>AttributeError</b>: 选择错误.

## get_initializer

获取使用的初始化器实例.

```python
get_initializer(initializer, seed)
```

### 参数

* <b>initializer</b>: 字符串或者```cml.initializers.Initializer``` 实例，决策树学习的划分方式.
* <b>seed</b>: 整数，初始化器的随机种子.

### 异常

* <b>AttributeError</b>: 模型编译的参数输入错误.

## get_kernel

获取使用的核函数实例.

```python
get_kernel(kernel, gamma)
```

### 参数

* <b>kernel</b>: 字符串或者```cml.kernels.Kernel``` 实例，核函数.
* <b>gamma</b>: 浮点数，核函数系数.

### 异常

* <b>AttributeError</b>: 模型编译的参数输入错误.

## get_loss

获取使用的损失函数实例.

```python
get_loss(loss)
```

### 参数

* <b>loss</b>: 字符串或者```cml.losses.Loss``` 实例，损失函数.

## get_metric

获取使用的评估函数实例.

```python
get_metric(metric)
```

### 参数

- **metric**: 字符串或者`cml.metrics.Metric` 实例，评估函数.

### 异常

- **AttributeError**: 模型编译的参数输入错误.

## get_optimizer

获取使用的优化器实例.

```python
get_optimizer(optimizer)
```

### 参数

- **optimizer**: 字符串或者`cml.optimizers.Optimizer` 实例，优化器.

### 异常

- **AttributeError**: 模型编译的参数输入错误.

## get_pruner

获取剪枝器.

```python
get_pruner(pruning)
```

### 参数

- **pruning**: 字符串，决策树剪枝的方式.

### 异常

- **AttributeError**: 选择错误.