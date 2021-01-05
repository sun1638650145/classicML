backend
=======

目前，
classicML标准版提供了\ ``CC``\ 和\ ``Python``\ 两个后端，如果你安装的是标准版，就可以根据你的需求切换后端。

1.切换你的后端
--------------

classicML
默认使用的是\ ``Python``\ 后端，环境变量\ ``CLASSICML_ENGINE``\ 的取值将决定使用的后端，如果你想切换后端，请输入：

.. code:: python

   import os
   os.environ['CLASSICML_ENGINE'] = 'CC'

我们尽可能的减少不同后端在代码层面产生的差异性，在你直接调用模型的过程中或者是直接调用\ ``cml.backend``\ 下的函数时，不同后端没有任何代码上的差异。你只会在你的终端中看到类似下面的提示：

.. code:: shell

   INFO:classicML:正在使用 CC 引擎
   INFO:classicML:后端版本是: backend.cc.ops.0.6

2.直接使用后端的算子
--------------------

在0.5版本之后，我们开放了使用后端算子的权限；在这种情况下，如果你想同时使用不同的后端提供的相同功能的函数，只要按照下面这样操作就可以了

.. code:: python

   from classicML.backend.cc.ops import cc_calculate_error  # CC后端的函数封装为python函数时，我们在函数名的前部增加cc_的标识
   from classicML.backend.python.ops import calculate_error

3.后端函数
----------

针对专业用户，你可以查看下面所列的全部函数和接口

.. toctree::
   :maxdepth: 1
   
   cc
   python/index
