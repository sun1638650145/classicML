# io

classicML的I/O函数，用于模型保存等操作.

## initialize_weights_file

初始化权重文件, 以创建或者解析模式运行.

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

