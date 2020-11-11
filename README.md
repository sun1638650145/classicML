# classicML

简单易用的经典机器学习框架，classicML支持数种机器学习算法，是你入门学习机器学习的首选

## 重要信息⚠️

1. 推荐你直接使用pip安装预编译的软件包(stable version)

   ```shell
   pip install classicML
   ```

2. 如果你有了比较高软件开发和编程水平可以从GitHub上下载源码进行编译安装

   ```
   git clone https://github.com/sun1638650145/classicML.git
   ```

3. 部分算法支持了使用CPP作为后端进行加速，你需要在开头使用这条语句切换后端

   ```python
   import os
   os.environ['CLASSICML_ENGINE'] = 'CC'
   ```
   
4. 使用CPP加速的版本是0.5alpha版本, 需要用户手动编译且目前只支持macOS和Linux暂不支持Windows.

5. 0.5版本的API接口略有改动, 修改了部分模块的路径, 结构更为合理.

6. 0.5版本添加benchmark模块可以监控内存和时间开销.

## 目前的已支持的算法

* 逻辑回归	
* 线性判别分析(添加CPP支持, 0.5alpha版本)
* BP神经网络
* 径向基函数神经网络
* 支持向量分类器(添加CPP支持, 0.5alpha版本)
* 分类决策树(添加CPP支持, 0.5alpha2版本)