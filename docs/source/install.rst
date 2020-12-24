.. _header-n59:

安装
====

.. warning::

    1. classicML 兼容的Python版本: Python 3.6-3.8
    2. classicML 仅支持64位的系统，强行使用32位系统可能会导致精度异常。
    3. classicML 目前仅支持x86-64架构的平台，arm架构暂不支持。

.. _header-n67:

1.在系统上安装Python开发环境
----------------------------

检查你是否安装了Python环境：

.. code:: shell

   python3 --version
   pip3 --version
   virtualenv --version

如果已经安装这些软件包，请直接跳过。

否则，请安装
`Python <https://www.python.org>`__\ 、\ `pip软件包管理器 <https://pip.pypa.io/en/stable/installing/>`__\ 、\ `virtualenv虚拟环境 <https://docs.python.org/zh-cn/3/library/venv.html>`__

.. _header-n72:

2.创建虚拟环境
--------------

Python 虚拟环境用于将软件包和系统隔离，避免你的误操作导致系统崩溃。

.. code:: shell

   python3 -m venv --system-site-packages ./classicML  # 如果你使用的是Windows操作系统，路径将修改为 .\classicML
   source ./classicML/bin/activate  # 激活虚拟环境

.. _header-n76:

3.安装classicML
---------------

使用PyPI安装预编译的软件包。

.. code:: shell

   pip install classicML-python  # 这个软件包使用纯Python编写的，没有额外的加速

.. _header-n87:

安装代码加速的软件包(可选)
--------------------------

.. warning::
    你执行完上面的操作就已经安装成功classicML，但是，你可以通过下面的方式安装拥有加速版本的classicML。你可以选择以下两种方式中的任意一种进行安装加速版的classicML。值得注意的是，PyPI预编译软件包仅适配了macOS和Linux；仅在理论上可以使用源码编译的方式在Windows安装，Windows未经过软件测试且作者长时间内不会针对适配。

.. _header-n136:

使用PyPI安装预编译的软件包
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   pip install classicML

.. _header-n132:

从GitHub上下载源代码进行构建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    1. Eigen 3 的版本不得低于 3.3.7
    2. pybind11 的版本不得低于 2.6
    3. C++ 的版本不得低于 c++14

.. _header-n99:

安装Eigen 3(以macOS为例)
^^^^^^^^^^^^^^^^^^^^^^^^

Eigen3 在不同平台软件包的名称可能有差异，安装方法也有差异。

.. code:: shell

   brew install eigen  # Linux下安装libeigen3-dev

.. _header-n103:

安装pybind11
^^^^^^^^^^^^

.. code:: python

   pip install pybind11

.. _header-n111:

下载源码并安装classicML
^^^^^^^^^^^^^^^^^^^^^^^

使用Git克隆仓库，安装脚本将自动安装classicML软件包。

.. code:: shell

   git clone https://github.com/sun1638650145/classicML.git
   cd classicML
   python3 setup.py index
