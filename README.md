# classicML 
[![Build Status](https://travis-ci.com/sun1638650145/classicML.svg?branch=master)](https://travis-ci.com/sun1638650145/classicML)

简单易用的经典机器学习框架，classicML支持数种机器学习算法，是你入门学习机器学习的首选

## 重要信息⚠️

1. 推荐你直接使用pip安装预编译的软件包(stable version)

   * 安装Python版本(没有加速)

     ```shell
     pip install classicML-python
     ```

   * 安装CPP版本(只支持macOS和Linux)

     ```shell
     pip install classicML
     ```

2. 如果你有了比较高软件开发和编程水平可以从GitHub上下载源码进行编译安装, 请预装

   * Eigen 3.3.7(setup.py里的eigen路径需要更改)
   * pybind 2.6+
   * 并且保证c++的版本最低版本为c++14

   ```shell
   git clone https://github.com/sun1638650145/classicML.git
   cd classicML
   python3 setup.py install
   ```

3. 部分算法支持了使用CPP作为后端进行加速，你需要在开头使用这条语句切换后端

   ```python
   import os
   os.environ['CLASSICML_ENGINE'] = 'CC'
   ```

4. 0.5版本的API接口略有改动, 修改了部分模块的路径, 结构更为合理.

5. 0.5版本添加benchmark模块可以监控内存和时间开销.

## 目前的已支持的算法

|      算法名称      | 支持多分类 | 使用CC加速 | 可视化 |      同时处理离散和连续值      |
| :----------------: | :--------: | :--------: | :----: | :----------------------------: |
|      逻辑回归      |            |            |   ✅    |                                |
|  线性判别分析LDA   |            |     ✅      |   ✅    |                                |
|     BP神经网络     |     ✅      |            |   ✅    | ✅ |
| 径向基函数神经网络 |            |            |   ✅    |                                |
| 支持向量分类器SVC  |            |     ✅      |   ✅    |                                |
|     分类决策树     |     ✅      |     ✅      |   ✅    |               ✅                |
| 朴素贝叶斯分类器NB |            |     ✅      | ✅ |               ✅                |
| 平均独依赖估计器AODE | | ✅ | | ✅ |
| 超父独依赖估计器SPODE | | ✅ | | ✅ |

* 全部神经网络只能可视化损失和评估函数曲线，暂不能可视化结构信息
* 其中BP神经网络需要手动将离散值转换成dummy编码