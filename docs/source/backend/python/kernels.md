# kernels

classicML的核函数.

## kernel

核函数的基类.

```python
cml.kernels.Kernel(name='kernel')
```

### 参数

* <b>name</b>: 字符串，核函数名称.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## Gaussian

高斯核函数. 具体实现参看径向基核函数.

```python
cml.kernels.Gaussian(name='gaussian', gamma=1.0)
```

### 参数

* <b>name</b>: 字符串，核函数名称.
* <b>gamma</b>: 浮点数，核函数系数.

## Linear

线性核函数.

```python
cml.kernels.Linear(name='linear')
```

### 参数

* <b>name</b>: 字符串，核函数名称.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 参数

* <b>x_i</b>: 一个 Numpy数组，第一组特征向量.
* <b>x_j</b>: 一个 Numpy数组，第二组特征向量.

#### 返回

核函数映射后的特征向量.

## Polynomial

多项式核函数.

```python
cml.kernels.Polynomial(name='poly', gamma=1.0, degree=3)
```

### 参数

* <b>name</b>: 字符串，核函数名称.
* <b>gamma</b>: 浮点数，核函数系数.
* <b>degree</b>: 整数，多项式的次数.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 参数

* <b>x_i</b>: 一个 Numpy数组，第一组特征向量.
* <b>x_j</b>: 一个 Numpy数组，第二组特征向量.

#### 返回

核函数映射后的特征向量.

## RBF

径向基核函数.

```python
cml.kernels.RBF(name='rbf', gamma=1.0)
```

### 参数

* <b>name</b>: 字符串，核函数名称.
* <b>gamma</b>: 浮点数，核函数系数.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 参数

* <b>x_i</b>: 一个 Numpy数组，第一组特征向量.
* <b>x_j</b>: 一个 Numpy数组，第二组特征向量.

#### 返回

核函数映射后的特征向量.

## Sigmoid

Sigmoid核函数.

```python
cml.kernels.Sigmoid(name='sigmoid', gamma=1.0, beta=1.0, theta=-1.0)
```

### 参数

* <b>name</b>: 字符串，核函数名称.
* <b>gamma</b>: 浮点数，核函数系数.
* <b>beta</b>: 浮点数，核函数参数.
* <b>theta</b>: 浮点数，核函数参数.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 参数

* <b>x_i</b>: 一个 Numpy数组，第一组特征向量.
* <b>x_j</b>: 一个 Numpy数组，第二组特征向量.

#### 返回

核函数映射后的特征向量.