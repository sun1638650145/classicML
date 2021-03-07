# classicML: 简单易用的经典机器学习框架

![build](https://github.com/sun1638650145/classicML/workflows/build/badge.svg) ![PyPI](https://github.com/sun1638650145/classicML/workflows/PyPI/badge.svg) [![Documentation Status](https://readthedocs.org/projects/classicml/badge/?version=latest)](https://classicml.readthedocs.io/zh_CN/latest/?badge=latest)

classicML 是一个用Python和C++混编的机器学习项目，它既实现了Python的简单易用快速上手，又实现了C++的高效性能。classicML的设计目标是简单易用，快速入门，编程风格简洁。

## 多后端支持

classicML 本身是一个Python项目，但是机器学习中涉及到的复杂的矩阵运算对于Python有点儿捉襟见肘，因此我们提供了使用C++后端的函数的加速版本。为了保证兼容性，classicML默认使用Python后端，部分算法支持了使用C++作为后端进行加速，你需要安装标准版的classicML，然后在开头使用这条语句切换后端。

```python
import os
os.environ['CLASSICML_ENGINE'] = 'CC'
```

## 第一个机器学习程序

使用线性判别分析进行二分类

* 下载示例数据集

```shell
wget https://github.com/sun1638650145/classicML/blob/master/datasets/西瓜数据集alpha.csv
```

* 运行下面的代码

```python
import pandas as pd
import classicML as cml

DATASET_PATH = '/path/to/西瓜数据集alpha.csv'

# 读取数据
dataframe = pd.read_csv(DATASET_PATH, index_col=0, header=0)
x = dataframe.iloc[:, :2].values
y = dataframe.iloc[:, 2].values
y[y == '是'] = 1
y[y == '否'] = 0
# 生成模型
model = cml.models.LDA()
# 训练模型
model.fit(x, y)
# 可视化模型
cml.plots.plot_lda(model, x, y, '密度', '含糖率')
```

* [更多示例代码点击](https://github.com/sun1638650145/classicML/tree/master/examples)
