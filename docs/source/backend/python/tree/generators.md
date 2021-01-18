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

一个```DecisionTreeClassifier```实例.