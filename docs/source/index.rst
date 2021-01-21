.. _header-n0:

classicML: 简单易用的经典机器学习框架
=====================================

|image0| |image1| |image2|

classicML
是一个用Python和CPP混编的机器学习项目，它既实现了Python的简单易用快速上手，又实现了CPP的高效性能. classicML的设计目标是简单易用，快速入门，编程风格简洁.

.. _header-n4:

多后端支持
----------

classicML
本身是一个Python项目，但是机器学习中涉及到的复杂的矩阵运算对于Python有点儿捉襟见肘，因此我们提供了使用CPP后端的函数的加速版本. 为了保证兼容性，classicML默认使用Python后端，部分算法支持了使用CPP作为后端进行加速，你需要安装标准版的classicML，然后在开头使用这条语句切换后端.

.. code:: python

   import os
   os.environ['CLASSICML_ENGINE'] = 'CC'

.. _header-n7:

第一个机器学习程序
------------------

使用线性判别分析进行二分类

-  下载示例数据集

.. code:: shell

   wget https://github.com/sun1638650145/classicML/blob/master/datasets/西瓜数据集alpha.csv

-  运行下面的代码

.. code:: python

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

-  `更多示例代码点击 <https://github.com/sun1638650145/classicML/tree/master/examples>`__

.. _header-n20:

目前的已支持的算法
------------------

classicML
目前支持数种机器学习算法，但是每种算法实现的情况有所不同和差异。

+----------+----------+----------+--------+----------+----------+
| 算法名称 | 支       | 使       | 可视化 | 同时     | 保存和   |
|          | 持多分类 | 用CC加速 |        | 处理离散 | 加载权重 |
|          |          |          |        | 和连续值 |          |
+==========+==========+==========+========+==========+==========+
| 逻辑回归 |          | ✅        | ✅      |          | ✅        |
+----------+----------+----------+--------+----------+----------+
| 线性     |          | ✅        | ✅      |          | ✅        |
| 判别分析 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| BP       | ✅        | ✅        | ✅      | ✅        | ✅        |
| 神经网络 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 径       |          | ✅        | ✅      |          | ✅        |
| 向基函数 |          |          |        |          |          |
| 神经网络 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 支持向   |          | ✅        | ✅      |          | ✅        |
| 量分类器 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 分       | ✅        | ✅        | ✅      | ✅        | ✅        |
| 类决策树 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 朴素贝叶 |          | ✅        | ✅      | ✅        | ✅        |
| 斯分类器 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 平均独依 |          | ✅        |        | ✅        | ✅        |
| 赖估计器 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+
| 超父独依 |          | ✅        | ✅      | ✅        | ✅        |
| 赖估计器 |          |          |        |          |          |
+----------+----------+----------+--------+----------+----------+

1. 全部神经网络只能可视化损失和评估函数曲线，暂不能可视化结构信息

2. BP神经网络需要手动将离散值转换成dummy编码

特别鸣谢
--------

首先，感谢每一个star和fork的用户，每一颗🌟就是对我最好的支持和奖励。然后，我要感谢在技术上给我各种帮助的人，感谢@Henry Schreiner先生，给我在Python打包时打包cc源码的指导；感谢@Daniel Saxton先生帮助我理解Pandas的源码。最后，感谢@Xin女士在文档编写上给予的意见和指导；感谢我的老姐@黎川女士帮助我进行的软件测试。

.. |image0| image:: https://github.com/sun1638650145/classicML/workflows/build/badge.svg
.. |image1| image:: https://github.com/sun1638650145/classicML/workflows/PyPI/badge.svg
.. |image2| image:: https://readthedocs.org/projects/classicml/badge/?version=latest
   :target: https://classicml.readthedocs.io/en/latest/?badge=latest
   
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Install
   
   install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API
   
   models
   plots
   backend/index
   benchmarks

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More
   
   FQA
   resources
   why-classicML
