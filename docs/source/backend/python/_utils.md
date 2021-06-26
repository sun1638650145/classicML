# _utils

classicML的工具类.

## ProgressBar

训练进度条.

```python
cml.backend.python._utils.ProgressBar(epochs, loss, metric)
```

### 参数

* <b>epochs</b>: 整数，训练的轮数.
* <b>loss</b>: 字符串或者```cml.losses.Loss```实例，模型使用的损失函数.
* <b>metric</b>: 字符串或者```cml.metrics.Metric``` 实例，模型使用的评估函数.

### \__call__

函数实现.

```python
__call__(epoch, current, loss_value, metric_value)
```

#### 参数

* <b>epoch</b>: 整数，当前的训练轮数.
* <b>current</b>: 浮点数，当前的时间戳.
* <b>loss_value</b>: 浮点数，当前的损失值.
* <b>metric_value</b>: 浮点数，当前的评估值.