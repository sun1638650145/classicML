# losses

classicML的损失函数.

## Loss

损失函数的基类.

```python
cml.losses.Loss(name=None)
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