## activations

classicML的激活函数.

### Activation

激活函数基类.

```python
cml.activations.Activation(name=None)
```

#### 参数

* <b>name</b>: 字符串，激活函数名称.

#### \__call__

函数实现.

```python
__call__(z)
```

##### 参数

* <b>z</b>: 一个Numpy数组，输入张量.

##### 异常

* <b>NotImplementedError</b>: 函数没有实现.

#### diff

函数的导数(微分).

```python
diff(output, a, *args, **kwargs)
```

##### 参数

* <b>output</b>: 一个Numpy数组，输出张量.
* <b>a</b>: 一个Numpy数组，输入张量.

##### 异常

* <b>NotImplementedError</b>: 函数没有实现.

### Relu

ReLU激活函数.

```python
cml.activations.Relu(name='relu')
```

#### 参数

* <b>name</b>: 字符串，激活函数名称.

#### \__call__

函数实现.

```python
__call__(z)
```

##### 参数

* <b>z</b>: 一个Numpy数组，输入张量.

##### 返回

经过激活后的张量.

#### diff

Relu的导数(微分).

```python
diff(output, a, *args, **kwargs)
```

##### 参数

* <b>output</b>: 一个Numpy数组，输出张量.
* <b>a</b>: 一个Numpy数组，输入张量.

##### 返回

Relu的导数(微分).

### Sigmoid

Sigmoid激活函数.

```python
cml.activations.Sigmoid(name='sigmoid')
```

#### 参数

* <b>name</b>: 字符串，激活函数名称.

#### \__call__

函数实现.

```python
__call__(z)
```

##### 参数

* <b>z</b>: 一个Numpy数组，输入张量.

##### 返回

经过激活后的张量.

#### diff

Sigmoid的导数(微分).

```python
diff(output, a, *args, **kwargs)
```

##### 参数

* <b>output</b>: 一个Numpy数组，输出张量.
* <b>a</b>: 一个Numpy数组，输入张量.
* <b>y_true</b>: numpy.ndarray，真实的标签.

##### 返回

Sigmoid的导数(微分).

### Softmax

Softmax激活函数.

```python
cml.activations.Softmax(name='softmax')
```

#### 参数

* <b>name</b>: 字符串，激活函数名称.

#### \__call__

函数实现.

```python
__call__(z)
```

##### 参数

* <b>z</b>: 一个Numpy数组，输入张量.

##### 返回

经过激活后的张量.

#### diff

Softmax的导数(微分).

```python
diff(output, a, *args, **kwargs)
```

##### 参数

* <b>output</b>: 一个Numpy数组，输出张量.
* <b>a</b>: 一个Numpy数组，输入张量.

##### 返回

Softmax的导数(微分).