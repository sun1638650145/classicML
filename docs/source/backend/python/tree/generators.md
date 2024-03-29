# generators

classicML中树结构的生成器.

## TreeGenerator

树生成器的基类.

```python
cml.backend.tree.generators(name=None, criterion=None)
```

### 参数

* <b>name</b>: 字符串，生成器的名称.
* <b>criterion</b>: {'gain', 'gini', 'entropy'}，决策树学习的划分方式.

###  \__call__

功能实现.

```python
__call__(*args, **kwargs)
```

### tree_generate

树的生成实现.

```python
tree_generate(*args, **kwargs)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## DecisionStumpGenerator

决策树桩生成器.

```python
cml.backend.tree.DecisionStumpGenerator(name='decision_stump_generator')
```

### 参数

* <b>name</b>: 字符串，生成器的名称.

###  \__call__

功能实现.

```python
__call__(*args, **kwargs)
```

### tree_generate

生成决策树桩.

```python
tree_generate(D, y)
```

#### 参数

* <b>D</b>: 一个Numpy数组, 特征数据.
* <b>y</b>: 一个Numpy数组, 标签.

#### 返回

`_DecisionStump`决策树桩实例.

### evaluate

对当前的划分方式进行评估.

```python
evaluate(D, y, column, dividing_point, division_mode)
```

#### 参数

* <b>D</b>: 一个Numpy数组, 特征数据.
* <b>y</b>: 一个Numpy数组, 标签.
* <b>column</b>: 整数, 划分的类别的下标.
* <b>dividing_point</b>: 浮点数, 划分点的值.
* <b>division_mode</b>: {'gte', 'le'}, 划分模式.

#### 返回

当前的划分方式评估的结果.

## DecisionTreeGenerator

决策树生成器.

```python
cml.backend.tree.DecisionTreeGenerator(name='decision_tree_generator', criterion=None)
```

### 参数

* <b>name</b>: 字符串，生成器的名称.
* <b>criterion</b>: {'gain', 'gini', 'entropy'}，决策树学习的划分方式.

###  \__call__

功能实现.

```python
__call__(*args, **kwargs)
```

### tree_generate

生成决策树.

```python
tree_generate(x, y)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.

#### 返回

`_TreeNode`树结点实例.

## TwoLevelDecisionTreeGenerator

2层决策树生成器.

```python
cml.backend.tree.TwoLevelDecisionTreeGenerator(name='2-level_decision_tree_generator', criterion='weighted_gini')
```

### 参数

* <b>name</b>: 字符串，生成器的名称.
* <b>criterion</b>: {'weighted_gini'}，2层决策树学习的划分方式.

###  \__call__

功能实现.

```python
__call__(*args, **kwargs)
```

### tree_generate

生成决策树.

```python
tree_generate(D, y, sample_distribution, height=0)
```

#### 参数

* <b>D</b>: 一个Numpy数组, 特征数据.
* <b>y</b>: 一个Numpy数组, 标签.
* <b>sample_distribution</b>: 一个Numpy数组, 样本分布.
* <b>height</b>: 整数, 决策树的高度.

#### 返回

`_TreeNode`树结点实例.

### choose_feature_to_divide

选择最优划分.

```python
choose_feature_to_divide(D, y, sample_distribution)
```

#### 参数

* <b>D</b>: 一个Numpy数组, 特征数据.
* <b>y</b>: 一个Numpy数组, 标签.
* <b>sample_distribution</b>: 一个Numpy数组, 样本分布.

#### 返回

当前结点划分属性的索引和划分点的值.