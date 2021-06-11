# io

classicML的I/O函数，用于模型保存等操作.

## initialize_weights_file

初始化权重文件, 以创建或者解析模式运行.

`cml.backend.io`的HDF5文件标准化协议包括: 两个信息 `description`和`parameters`, `description`用来存放cml兼容性和开发时间等信息; `parameters`用以保存模型本身的参数.`parameters`分成两个数据集: `compile`保存模型的训练的超参数; `weights`保存模型的权重信息. 开发符合标准化协议的自定义模型, 需将固化的参数保存在`compile`和`weights`中.

```python
cml.io.initialize_weights_file(filepath, mode, model_name)
```

### 参数

* <b>filepath</b>: 字符串，权重文件加载的路径.
* <b>mode</b>: ```'w'```或``` 'r'```，工作模式，```'w'```是写入权重文件，```'r'```是读取权重文件.
* <b>model_name</b>: 字符串，模型的名称.

### 返回

* 可操作的文件指针.

### 异常

* <b>IOError</b>: 初始化失败.
* <b>ValueError</b>: 文件核验失败.

