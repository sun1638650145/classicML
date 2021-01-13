# initializers

classicML的初始化函数.

## Initializer

初始化函数基类.

```python
cml.initializers.Initializer(name=None, seed=None)
```

### 参数

* <b>name</b>: 字符串，激活函数名称.
* <b>seed</b>: 整数，初始化器的随机种子.

### \__call__

函数实现.

```python
__call__(*args, **kwargs)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## RandomNormal

正态分布随机初始化器.

```python
cml.initializers.RandomNormal(name='random_normal', seed=None)
```

### 参数

* <b>name</b>: 字符串，激活函数名称.
* <b>seed</b>: 整数，初始化器的随机种子.

### \__call__

函数实现.

```python
__call__(attributes_or_structure)
```

#### 参数

* <b>attributes_or_structure</b>: 整数或列表, 如果是逻辑回归就是样本的特征数; 如果是神经网络, 就是定义神经网络的网络结构

