# optimizers

classicML的优化器.

## Optimizer

优化器的基类.

```python
cml.optimizer.Optimizer(name=None)
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
cml.optimizer.Adam(name='adam',
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

