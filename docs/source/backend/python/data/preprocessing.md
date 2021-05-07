# preprocessing

classicML中的数据预处理模块.

## PreProcessor

预处理器基类，预处理器将实现一系列预处理操作，部分预处理器还有对应的逆操作.

```python
cml.data.preprocessing.PreProcessor(name='preprocessor')
```

### 参数

* <b>name</b>: 字符串，预处理器名称.

### \__call__

预处理操作.

```python
__call__(*args, **kwargs)
```

#### 异常

* <b>NotImplementedError</b>: \__call__方法需要用户实现..

### inverse

预处理逆操作.

```python
inverse(*args, **kwargs)
```

#### 异常

* <b>NotImplemented</b>: inverse方法需要用户实现.

## DummyEncoder

Dummy编码器.

```python
cml.data.preprocessing.DummyEncoder(name='dummy_encoder', dtype='float32')
```

### 参数

* <b>name</b>: 字符串，Dummy编码器名称.
* <b>dtype</b>: 字符串，编码后的标签的数据类型.

### \__call__

进行Dummy编码.

```python
__call__(labels)
```

#### 参数

* <b>labels</b>: 一个Numpy数组，原始的标签.

#### 返回

Dummy编码后的标签.

## Imputer

缺失值填充器，连续值将填充均值，离散值将填充众数.

```python
cml.data.preprocessing.Imputer(name='imputer')
```

### 参数

* <b>name</b>: 字符串，缺失值填充器名称.

### \__call__

进行缺失值填充.

```python
__call__(data)
```

#### 参数

* <b>data</b>: 一个Numpy数组，输入的数据.

#### 返回

填充后的数据.

## MaxMarginEncoder

最大化间隔编码器, 对于支持向量机的标签编码需要将编码转换为关于超平面的.

```python
cml.data.preprocessing.MaxMarginEncoder(name='max_margin_encoder', dtype='float32')
```

### 参数

* <b>name</b>: 字符串，最大化间隔编码器名称.
* <b>dtype</b>: 字符串，编码后的标签的数据类型.

### \__call__

进行最大化间隔编码.

```python
__call__(labels)
```

#### 参数

* <b>labels</b>: 一个Numpy数组，原始的标签.

#### 返回

最大化间隔编码后的标签，类标签和类索引的映射字典.

## MinMaxScaler

归一化器.

```python
cml.data.preprocessing.MinMaxScaler(name='minmax_scalar', dtype='float32', axis=-1)
```

### 参数

* <b>name</b>: 字符串，归一化器的名称.
* <b>dtype</b>: 字符串，编码后的标签的数据类型.
* <b>axis</b>: 整数，归一化所沿轴.

### \__call__

进行归一化.

```python
__call__(data)
```

#### 参数

* <b>data</b>: 一个Numpy数组，输入的数据.

#### 返回

归一化后的数据.

### inverse

进行反归一化.

```python
inverse(preprocessed_data)
```

#### 参数

* <b>preprocessed_data</b>: 一个Numpy数组，输入的归一化后数据.

#### 返回

归一化前的数据.

## OneHotEncoder

独热编码器.

```python
cml.data.preprocessing.OneHotEncoder(name='one-hot_encoder', dtype='float32')
```

### 参数

* <b>name</b>: 字符串，独热编码器的名称.
* <b>dtype</b>: 字符串，编码后的标签的数据类型.

### \__call__

进行独热编码.

```python
__call__(labels)
```

#### 参数

* <b>labels</b>: 一个Numpy数组，原始的标签.

#### 返回

独热编码后的标签，类标签和类索引的映射字典.

## StandardScaler

标准化器.

```python
cml.data.preprocessing.StandardScaler(name='standard_scalar', dtype='float32', axis=-1
```

### 参数

* <b>name</b>: 字符串，标准化器的名称.
* <b>dtype</b>: 字符串，标准化后数据元素的数据类型.
* <b>axis</b>: 整数，标准化所沿轴.

### \__call__

进行标准化.

```python
__call__(data)
```

#### 参数

* <b>data</b>: 一个Numpy数组，输入的数据.

#### 返回

标准化后的数据.

### inverse

进行反标准化.

```python
inverse(preprocessed_data)
```

#### 参数

* <b>preprocessed_data</b>: 一个Numpy数组，输入的标准化后数据.

#### 返回

标准化前的数据.