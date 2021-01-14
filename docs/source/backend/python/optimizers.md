# optimizers

classicML的优化器.

## Optimizer

优化器的基类.

```python
cml.optimizers.Optimizer(name=None)
```

### 参数

* <b>name</b>: 字符串，优化器的名称.

### \__call__

函数实现.

```python
__call__(x, y, epochs, parameters, *args, **kwargs)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>y</b>:一个 Numpy 数组，标签.
* <b>epochs</b>:整数，训练的轮数.
* <b>parameters</b>:一个 Numpy 数组，模型的参数矩阵.

### run

运行优化器优化参数.

```python
run(x, y, epochs, parameters, *args, **kwargs)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>y</b>: 一个 Numpy 数组，标签.
* <b>epochs</b>: 整数，训练的轮数.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.
* <b>verbose</b>: 布尔值，显示日志信息.
* <b>loss</b>: 字符串或者```cml.losses.Loss```实例，模型使用的损失函数.
* <b>metric</b>: 字符串或者```cml.metrics.Metric```实例，模型使用的评估函数.
* <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

模型的参数矩阵.

## Adam

自适应矩估计优化器.

```python
cml.optimizers.Adam(name='adam',
                   learning_rate=1e-3,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-7)
```

### 参数

* <b>name</b>: 字符串，优化器的名称.
* <b>learning_rate</b>: 浮点数，优化器的学习率.
* <b>beta_1</b>: 浮点数，一阶矩估计衰减率.
* <b>beta_2</b>: 浮点数，二阶矩估计衰减率.
* <b>epsilon</b>:浮点数，数值稳定的小常数.

### run

运行优化器优化参数.

```python
run(x, y, epochs, parameters, *args, **kwargs)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>y</b>: 一个 Numpy 数组，标签.
* <b>epochs</b>: 整数，训练的轮数.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.
* <b>verbose</b>: 布尔值，显示日志信息.
* <b>loss</b>: 字符串或者```cml.losses.Loss```实例，模型使用的损失函数.
* <b>metric</b>: 字符串或者```cml.metrics.Metric```实例，模型使用的评估函数.
* <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

模型的参数矩阵.

## GradientDescent

梯度下降优化器.

```python
cml.optimizers.GradientDescent(name='gradient_descent',
                              learning_rate=1e-2)
```

### 参数

- **name**: 字符串，优化器的名称.
- **learning_rate**: 浮点数，优化器的学习率.

### run

运行优化器优化参数.

```
run(x, y, epochs, beta, *args, **kwargs)
```

#### 参数

- **x**: 一个 Numpy 数组，特征数据.
- **y**: 一个 Numpy 数组，标签.
- **epochs**: 整数，训练的轮数.
- **beta**: 一个 Numpy 数组，模型的参数矩阵.
- **verbose**: 布尔值，显示日志信息.
- **loss**: 字符串或者`cml.losses.Loss`实例，模型使用的损失函数.
- **metric**: 字符串或者`cml.metrics.Metric`实例，模型使用的评估函数.

#### 返回

模型的参数矩阵.

### forward

优化器前向传播.

```python
cml.optimizers.GradientDescent.forward(x, parameters)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.

#### 返回

预测的标签(概率形式)和参数(x;1)矩阵.

### backward

优化器反向传播.

```python
cml.optimizers.GradientDescent.backward(y_pred, y_true, x_hat)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy 数组，预测的标签(概率形式).
* <b>y_true</b>: 一个 Numpy 数组，真实的标签.
* <b>x_hat</b>: 一个 Numpy 数组，属性的参数矩阵.

#### 返回

优化器的实时梯度矩阵.

## NewtonMethod

牛顿法优化器.

```python
cml.optimizers.NewtonMethod(name='newton_method')
```

### 参数

- **name**: 字符串，优化器的名称.

### run

运行优化器优化参数.

```
run(x, y, epochs, beta, *args, **kwargs)
```

#### 参数

- **x**: 一个 Numpy 数组，特征数据.
- **y**: 一个 Numpy 数组，标签.
- **epochs**: 整数，训练的轮数.
- **beta**: 一个 Numpy 数组，模型的参数矩阵.
- **verbose**: 布尔值，显示日志信息.
- **loss**: 字符串或者`cml.losses.Loss`实例，模型使用的损失函数.
- **metric**: 字符串或者`cml.metrics.Metric`实例，模型使用的评估函数.

#### 返回

模型的参数矩阵.

### forward

优化器前向传播.

```python
cml.optimizers.NewtonMethod.forward(x, parameters)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.

#### 返回

预测的标签(概率形式)和参数(x;1)矩阵.

### backward

优化器反向传播.

```python
cml.optimizers.NewtonMethod.backward(y_pred, y_true, x_hat)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy 数组，预测的标签(概率形式).
* <b>y_true</b>: 一个 Numpy 数组，真实的标签.
* <b>x_hat</b>: 一个 Numpy 数组，属性的参数矩阵.

#### 返回

优化器的实时梯度矩阵.

## RadialBasisFunctionOptimizer

径向基函数优化器.

```python
cml.optimizers.RadialBasisFunctionOptimizer(name='rbf',
                                            learning_rate=1e-2)
```

### 参数

- **name**: 字符串，优化器的名称.
- **learning_rate**: 浮点数，优化器的学习率.

### run

运行优化器优化参数.

```
run(x, y, epochs, parameters, *args, **kwargs)
```

#### 参数

- **x**: 一个 Numpy 数组，特征数据.
- **y**: 一个 Numpy 数组，标签.
- **epochs**: 整数，训练的轮数.
- **parameters**: 一个 Numpy 数组，模型的参数矩阵.
- **verbose**: 布尔值，显示日志信息.
- **loss**: 字符串或者`cml.losses.Loss`实例，模型使用的损失函数.
- **metric**: 字符串或者`cml.metrics.Metric`实例，模型使用的评估函数.
- <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

模型的参数矩阵.

### forward

优化器前向传播.

```python
cml.optimizers.RadialBasisFunctionOptimizer.forward(x, parameters)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.

#### 返回

预测的标签(概率形式)和参数矩阵缓存.

### backward

优化器反向传播.

```python
cml.optimizers.RadialBasisFunctionOptimizer.backward(y_pred, y_true, cache)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy 数组，预测的标签(概率形式).
* <b>y_true</b>: 一个 Numpy 数组，真实的标签.
* <b>cache</b>: 一个 Numpy 数组，参数缓存.

#### 返回

优化器的实时梯度矩阵.

## SequentialMinimalOptimization

序列最小最优化算法. SMO算法是一种启发式算法，即每次优化两个变量，使之满足KKT条件；不断迭代，最后使得全部变量满足KKT条件. 整个SMO算法包括：求解两个变量的二次规划问题和选择变量的启发式方法.

```python
cml.optimizers.SequentialMinimalOptimization(name='SMO') # 可以使用缩写 cml.models.SMO()
```

### 参数

- **name**: 字符串，优化器的名称.

### run

运行优化器优化参数.

```
run(x, y, *args, **kwargs)
```

#### 参数

- **x**: 一个 Numpy 数组，特征数据.
- **y**: 一个 Numpy 数组，标签.
- **C**: 浮点数，软间隔正则化系数.
- **kernel**: 字符串或者```cml.kernels.Kernel```，分类器使用的核函数.
- **tol**: 浮点数，停止训练的最大误差值.
- **epochs**: 整数，最大的训练轮数，如果是-1则表示需要所有的样本满足条件时，
  才能停止训练，即没有限制.

#### 返回

分类器的支持向量下标数组, 支持向量数组, 拉格朗日乘子数组, 支持向量对应的标签数组和偏置项.

## StochasticGradientDescent

随机梯度下降优化器.

```python
cml.optimizers.StochasticGradientDescent(name='stochastic_gradient_descent',
                                         learning_rate=1e-2) # 可以使用缩写 cml.models.SGD()
```

### 参数

- **name**: 字符串，优化器的名称.
- **learning_rate**: 浮点数，优化器的学习率.

### 注意

* 如果想固定随机种子，实现复现的话，请在模型实例化的时候将随机种子置为一个常整数.

### run

运行优化器优化参数.

```
run(x, y, epochs, parameters, *args, **kwargs)
```

#### 参数

- **x**: 一个 Numpy 数组，特征数据.
- **y**: 一个 Numpy 数组，标签.
- **epochs**: 整数，训练的轮数.
- **parameters**: 一个 Numpy 数组，模型的参数矩阵.
- **verbose**: 布尔值，显示日志信息.
- **loss**: 字符串或者`cml.losses.Loss`实例，模型使用的损失函数.
- **metric**: 字符串或者`cml.metrics.Metric`实例，模型使用的评估函数.
- <b>callbacks</b>: 列表，模型训练过程的中间数据记录器.

#### 返回

模型的参数矩阵.

### forward

优化器前向传播.

```python
cml.optimizers.StochasticGradientDescent.forward(x, parameters)
```

#### 参数

* <b>x</b>: 一个 Numpy 数组，特征数据.
* <b>parameters</b>: 一个 Numpy 数组，模型的参数矩阵.

#### 返回

预测的标签(概率形式)和参数矩阵缓存.

### backward

优化器反向传播.

```python
cml.optimizers.StochasticGradientDescent.backward(y_pred, y_true, cache)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy 数组，预测的标签(概率形式).
* <b>y_true</b>: 一个 Numpy 数组，真实的标签.
* <b>cache</b>: 一个 Numpy 数组，参数缓存.

#### 返回

优化器的实时梯度矩阵字典.