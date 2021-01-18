# pruners

classicML中的树的剪枝器.

## Pruner

剪枝器基类.

```python
cml.backend.tree.pruners.Pruner(name=None)
```

### 参数

* <b>name</b>: 字符串，剪枝器的名称.

### \__call__

进行剪枝操作.

```python
__call__(x, y, x_validation, y_validation, tree)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.
* <b>x_validation</b>: Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: Pandas的DataFrame，剪枝使用的验证标签.
* <b>tree</b>: ```cml.backend.tree._TreeNode```实例，决策树.

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

### calculation_accuracy

计算使用预后剪枝操作的之后前的准确率.

```python
calculation_accuracy(*args, **kwargs):
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

## PostPruner

后剪枝器.

```python
cml.backend.tree.pruners.PostPruner(name='post')
```

### 参数

* <b>name</b>: 字符串，剪枝器的名称.

### \__call__

进行剪枝操作.

```python
__call__(x, y, x_validation, y_validation, tree)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.
* <b>x_validation</b>: Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: Pandas的DataFrame，剪枝使用的验证标签.
* <b>tree</b>: ```cml.backend.tree._TreeNode```实例，决策树.

### calculation_accuracy

计算剪枝前的准确率. 

这里没有采取原文的做法，而是只计算这一个分支的数据，因为不修剪其他分支，其他分支当前的值不改变也就不会影响准确率的总体变化，这样不仅代码好实现，而且同时显著减少计算的开销.

```python
calculation_accuracy(x_validation, y_validation, tree)
```

#### 参数

* <b>x_validation</b>: Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: Pandas的DataFrame，剪枝使用的验证标签.
* <b>tree</b>: ```cml.backend.tree._TreeNode```实例，决策树.

## PrePruner

预剪枝器.

```python
cml.backend.tree.pruners.PrePruner(name='pre')
```

### 参数

* <b>name</b>: 字符串，剪枝器的名称.

### 注意

* 这里只取用了预剪枝算法的思想，实际实现还是在决策树生成以后进行的剪枝操作，因为如果按照原文实现势必影响一个正常的决策树生成.

### \__call__

进行剪枝操作.

```python
__call__(x, y, x_validation, y_validation, tree)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.
* <b>x_validation</b>: Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: Pandas的DataFrame，剪枝使用的验证标签.
* <b>tree</b>: ```cml.backend.tree._TreeNode```实例，决策树.

### calculation_accuracy

计算预剪枝划分后的准确率.

```python
calculation_accuracy(x, y, x_validation, y_validation, tree)
```

#### 参数

* <b>x</b>: Pandas的DataFrame，特征数据.
* <b>y</b>: Pandas的DataFrame，标签.
* <b>x_validation</b>: Pandas的DataFrame，剪枝使用的验证特征数据.
* <b>y_validation</b>: Pandas的DataFrame，剪枝使用的验证标签.
* <b>tree</b>: ```cml.backend.tree._TreeNode```实例，决策树.