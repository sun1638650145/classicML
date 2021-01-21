.. _header-n3:

cc
==

``cc``\ 后端将逐步重写\ ``python``\ 后端的全部函数和接口，并针对时间和内存开销进行大幅度的优化。\ ``cc``\ 后端暴露出来的函数将以\ ``cc_xxx_function``\ 与原函数进行区分，类暂时和原\ ``Python``\ 的类名一致，不同时调用多后端函数请忽略.

.. toctree::
   :maxdepth: 2

   activations
   metrics
   ops
