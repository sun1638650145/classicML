.. _header-n3:

cc
==
    
.. note::

    1. ``cc``\ 后端目前重写了接近一半的\ ``python``\ 后端，并针对时间开销进行大幅度的优化，平均优化幅度接近20%(基于作者的测试).
    2. ``cc``\ 后端的函数将以\ ``cc_xxx_function``\ 与原函数进行区分，类和\ ``Python``\ 后端类名一致，不同时调用多后端函数请忽略此提示.

.. toctree::
   :maxdepth: 2
	
   _utils
   activations
   callbacks
   initializers
   kernels
   losses
   metrics
   ops
