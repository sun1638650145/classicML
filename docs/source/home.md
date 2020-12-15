# classicML: 简单易用的经典机器学习框架

![build](https://github.com/sun1638650145/classicML/workflows/build/badge.svg) ![PyPI](https://github.com/sun1638650145/classicML/workflows/PyPI/badge.svg) [![Documentation Status](https://readthedocs.org/projects/classicml/badge/?version=latest)](https://classicml.readthedocs.io/en/latest/?badge=latest)

classicML是一个用Python和CPP混编的机器学习项目，它的设计目标是简单易用，快速入门，编程风格简洁。

## 第一步

### 使用PyPI安装预编译的软件包

注意：PyPI提供了macOS和Linux的预编译CPP加速包，和macOS、Linux和Windows的预编译Python包

* 安装Python版本

```shell
pip install classicML-python
```

* 安装CPP版本

```shell
pip install classicML
```

### 从GitHub上下载源码进行编译安装

注意：从源码安装将默认安装CPP版本，Windows仅在理论上支持未经过软件测试且作者长时间内不会针对适配

* 安装依赖的CPP库(以macOS为例)

```shell
brew install eigen
```

* 安装依赖的Python库

```python
pip install pybind11
```

* 使用git克隆仓库

```shell
git clone https://github.com/sun1638650145/classicML.git
```

* 进行安装

```shell
cd classicML
python3 setup.py install
```

### 重要信息⚠️

* classicML兼容的Python版本: Python 3.6-3.8
* classicML仅支持64位的系统，强行使用32位系统可能会导致精度异常
* classicML要求的Eigen版本是 3.3.7+
* classicML要求的pybind版本是 2.6+
* 编译使用的c++版本不能低于 c++14

## 第一行

* 使用线性判别分析

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

* 下载示例数据集

```shell
wget https://github.com/sun1638650145/classicML/blob/master/datasets/西瓜数据集alpha.csv
```

* [更多示例代码点击](https://github.com/sun1638650145/classicML/tree/master/examples)

## 切换后端

为了保证兼容性classicML默认使用Python后端，部分算法支持了使用CPP作为后端进行加速，你需要在开头使用这条语句切换后端，或者直接修改项目文件中的预设环境变量(不推荐)

```python
import os
os.environ['CLASSICML_ENGINE'] = 'CC'
```

## 目前的已支持的算法

|      算法名称      | 支持多分类 | 使用CC加速 | 可视化 | 同时处理离散和连续值 |
| :----------------: | :--------: | :--------: | :----: | :------------------: |
|      逻辑回归      |            |            |   ✅    |                      |
|    线性判别分析    |            |     ✅      |   ✅    |                      |
|     BP神经网络     |     ✅      |            |   ✅    |          ✅           |
| 径向基函数神经网络 |            |            |   ✅    |                      |
|   支持向量分类器   |            |     ✅      |   ✅    |                      |
|     分类决策树     |     ✅      |     ✅      |   ✅    |          ✅           |
|  朴素贝叶斯分类器  |            |     ✅      |   ✅    |          ✅           |
|  平均独依赖估计器  |            |     ✅      |        |          ✅           |
|  超父独依赖估计器  |            |     ✅      |   ✅    |          ✅           |

1. 全部神经网络只能可视化损失和评估函数曲线，暂不能可视化结构信息

2. 其中BP神经网络需要手动将离散值转换成dummy编码

