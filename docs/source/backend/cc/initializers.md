# initializers

classicML的初始化器.

## Initializer

初始化函数基类.

```python
cml.initializers.Initializer(name='initializer', seed=None)
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

## GlorotNormal

Glorot正态分布随机初始化器，具体实现参看XavierNormal. Xavier正态分布随机初始化器，也叫做Glorot正态分布随机初始化器.

```python
cml.initializers.GlorotNormal(name='glorot_normal', seed=None)
```

### 参数

- <b>name</b>: 字符串，激活函数名称.
- <b>seed</b>: 整数，初始化器的随机种子.

## HeNormal

He正态分布随机初始化器.

```python
cml.initializers.HeNormal(name='he_normal', seed=None)
```

### 参数

* <b>name</b>: 字符串，激活函数名称.
* <b>seed</b>: 整数，初始化器的随机种子.

###  参考文献

* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) 

### \__call__

初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

```python
__call__(attributes_or_structure)
```

#### 参数

* <b>attributes_or_structure</b>: 整数或列表，如果是逻辑回归就是样本的特征数；如果是神经网络，就是定义神经网络的网络结构.

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

* <b>attributes_or_structure</b>: 整数或列表，如果是逻辑回归就是样本的特征数；如果是神经网络，就是定义神经网络的网络结构.

## RBFNormal

RBF网络的初始化器.

```python
cml.initializers.RBFNormal(name='rbf_normal', seed=None)
```

### 参数

* <b>name</b>: 字符串，激活函数名称.
* <b>seed</b>: 整数，初始化器的随机种子.

### \__call__

函数实现.

```python
__call__(hidden_units)
```

#### 参数

* <b>hidden_units</b>: 整数，径向基函数网络的隐含层神经元数量.

#### 注意

* 这里隐含层神经元中心本应用```np.random.randn```全部初始化，但是实际工程发现，有负值的时候可能会导致求高斯函数的时候增加损失不收敛，因此，全部初始化为正数.

## XavierNormal

Xavier正态分布随机初始化器，也叫做Glorot正态分布随机初始化器.

```python
cml.initializers.XavierNormal(name='xavier_normal', seed=None)
```

### 参数

* <b>name</b>: 字符串，激活函数名称.
* <b>seed</b>: 整数，初始化器的随机种子.

###  参考文献

* [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)

### \__call__

初始化方式为W~N(0, sqrt(2/N_in+N_out))，其中N_in为对应连接的输入层的神经元个数，N_out为本层的神经元个数.

```python
__call__(attributes_or_structure)
```

#### 参数

* <b>attributes_or_structure</b>: 整数或列表，如果是逻辑回归就是样本的特征数；如果是神经网络，就是定义神经网络的网络结构.