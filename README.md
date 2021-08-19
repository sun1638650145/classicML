# classicML: 简单易用的经典机器学习框架

![build](https://github.com/sun1638650145/classicML/workflows/build/badge.svg) ![PyPI](https://github.com/sun1638650145/classicML/workflows/PyPI/badge.svg) [![Documentation Status](https://readthedocs.org/projects/classicml/badge/?version=latest)](https://classicml.readthedocs.io/zh_CN/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/classicML?color=blue)](https://pypi.org/project/classicML/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/classicML)](https://pypi.org/project/classicML/)

classicML 是一个用 Python 和 C++ 混编的机器学习项目，您既可以使用纯 Python 版本进行学习，也可以使用CC标准版进行实验和探索自定义功能。它既实现了Python的简单易用快速上手，又实现了C++的高效性能。classicML的设计目标是简单易用，快速入门，高扩展性和编程风格简洁。更多信息请访问[文档网站](https://classicml.readthedocs.io/)。

## 多后端支持

classicML 本身是一个Python项目，但是机器学习中涉及到的复杂的矩阵运算对于Python有点儿捉襟见肘，因此我们提供了使用C++后端的加速版本。为了保证兼容性，classicML默认使用Python后端，现在全部算法支持了使用C++作为后端进行加速，如果您需要使用标准版的classicML，只需在开头使用这条语句切换后端。

```python
import os
os.environ['CLASSICML_ENGINE'] = 'CC'
```

## 精度控制

目前，classicML 正在对全部算法支持32位和64位切换精度，使用32位的精度可以获得更快的运行速度和更小固化模型。

```python
import os
os.environ['CLASSICML_PRECISION'] = '32-bit'
```

## 第一个机器学习程序

使用线性判别分析进行二分类

* 下载示例数据集

```shell
wget https://github.com/sun1638650145/classicML/blob/master/datasets/西瓜数据集alpha.csv
```

* 运行下面的代码

```python
import classicML as cml

DATASET_PATH = '/path/to/西瓜数据集alpha.csv'

# 读取数据
ds = cml.data.Dataset()
ds.from_csv(DATASET_PATH)
# 生成模型
model = cml.models.LDA()
# 训练模型
model.fit(ds.x, ds.y)
# 可视化模型
cml.plots.plot_lda(model, ds.x, ds.y, '密度', '含糖率')
```

* [更多示例代码点击](https://github.com/sun1638650145/classicML/tree/master/examples)

## v0.7 预览

在之前的版本`classicML`全局精度是`float64`的，现在我们引入`CLASSICML_PRECISION`，这样你就可以控制全局精度，使用32位的精度时，可以获得更快的运行速度和更小固化模型。

