# criteria

classicML中决策树的划分标准.

## Criterion

划分标准基类.

```python
cml.backend.tree.criteria.Criterion(name=None)
```

### 参数

* <b>name</b>: 字符串，划分标准的名称.

### \__call__

划分标准算法实现.

```python
__call__(D)
```

#### 参数

* <b>D</b>: Pandas的Series，需要计算的数据集.

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

### get_value

计算划分标准的值.

```python
get_value(*args, **kwargs)
```

#### 参数

- <b>D</b>: Pandas的Series，需要计算的数据集.
- <b>y</b>: Pandas的DataFrame，对应的标签.
- <b>continuous:</b> 布尔值, 是否是连续属性.

### optimal_division

最优的划分属性.

```python
optimal_division(x, y)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.

## Entropy

信息熵.

```python
cml.backend.tree.criteria.Entropy(name='entropy')
```

### 参数

* <b>name</b>: 字符串，划分标准的名称.

### \__call__

计算信息熵.

```python
__call__(D)
```

#### 参数

* <b>D</b>: Pandas的Series，需要计算的数据集.

## Gain

信息增益.

```python
cml.backend.tree.criteria.Gain(name='gain')
```

### 参数

* <b>name</b>: 字符串，划分标准的名称.

### get_value

计算信息增益.

```python
get_value(D, y, D_entropy, continuous)
```

#### 参数

- <b>D</b>: Pandas的Series，需要计算的数据集.
- <b>y</b>: Pandas的DataFrame，对应的标签.
- <b>D_entropy</b>: 浮点数，整个数据集的信息熵.
- <b>continuous:</b> 布尔值, 是否是连续属性.

### optimal_division

最优的划分属性.

```python
optimal_division(x, y)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.

## Gini

基尼指数.

```python
cml.backend.tree.criteria.Gini(name='gini')
```

### 参数

* <b>name</b>: 字符串，划分标准的名称.

### \__call__

计算基尼指数.

```python
__call__(D)
```

#### 参数

* <b>D</b>: Pandas的Series，需要计算的数据集.

## WeightedGini

带权重的基尼指数.

```python
cml.backend.tree.criteria.Gini(name='weighted_gini')
```

### 参数

* <b>name</b>: 字符串，划分标准的名称.

### \__call__

计算带权重的基尼指数.

```python
__call__(D, sample_distribution)
```

#### 参数

* <b>D</b>: 一个 Numpy数组，需要计算的数据集.
* <b>sample_distribution</b>: 一个 Numpy数组，样本分布.

### get_value

计算基尼指数的值.

```python
get_value(*args, **kwargs)
```

#### 参数

- <b>D</b>: 一个 Numpy数组，需要计算的数据集.

- <b>y</b>: 一个 Numpy数组，对应的标签.

- <b>sample_distribution</b>: 一个 Numpy数组，样本分布.

  