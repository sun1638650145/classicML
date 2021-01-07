## callbacks

classicML的回调函数，可以使用回调函数来查看训练的历史记录.

### History

保存训练的历史记录.

```python
cml.callbacks.History(name=None, loss_name='loss', metric_name='metric')
```

#### 参数

* <b>name</b>: 字符串，历史记录的名称.
* <b>loss_name</b>: 字符串，使用损失函数的名称.
* <b>metric_name</b>: 字符串，使用评估函数的名称.

#### \__call__

记录当前的信息.

```python
__call__(loss_value, metric_value)
```

##### 参数

* <b>loss_value</b>: 浮点数，当前的损失值.
* <b>metric_value</b>: 浮点数，当前的评估值.