# losses

classicML的损失函数.

## Loss

损失函数的基类.

```python
cml.losses.Loss(name='loss')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## BinaryCrossentropy

二分类交叉熵损失函数.

```python
cml.losses.BinaryCrossentropy(name='binary_crossentropy')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的损失值.

## CategoricalCrossentropy

多分类交叉熵损失函数.

```python
cml.losses.CategoricalCrossentropy(name='categorical_crossentropy')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的损失值.

## Crossentropy

交叉熵损失函数，将根据标签的实际形状自动使用二分类或者多分类损失函数.

```python
cml.losses.Crossentropy(name='crossentropy')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的损失值.

## LogLikelihood

对数似然损失函数.

```python
cml.losses.CategoricalCrossentropy(name='log_likelihood')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_true, beta, *args, **kwargs)
```

#### 参数

* <b>y_true</b>: 一个 Numpy数组，真实的标签.
* <b>beta</b>: 一个 Numpy数组，模型的参数矩阵.
* <b>x_hat</b>: 一个 Numpy数组，属性的参数矩阵.

#### 返回

当前的损失值.

## MeanSquaredError

均方误差损失函数.

```python
cml.losses.MeanSquaredError(name='mean_squared_error')
```

### 参数

* <b>name</b>: 字符串，损失函数名称.

### \__call__

函数实现.

```python
__call__(y_pred, y_true, *args, **kwargs)
```

#### 参数

* <b>y_pred</b>: 一个 Numpy数组，预测的标签.
* <b>y_true</b>: 一个 Numpy数组，真实的标签.

#### 返回

当前的损失值.

