# benchmarks

classicML 中的```benchmarks```用于评估和测试模型的性能和开销，借此，你可以分析性能瓶颈；目前以装饰器的形式使用，比如：

```python
import classicML as cml

@cml.benchmarks.timer
def create_model():
    """创建一个模型"""
    model = cml.models.BPNN(seed=2020)
    return model

if __name__ == '__main__':
    create_model()
```

在这个例子中，你可以查看模型的创建速度.

```shell
INFO:classicML:正在使用 Python 引擎
INFO:classicML:耗时 0.00001 s
```

第二个例子可以查看，整个进程所占用的内存，以此，找到优化的策略.

```python
import classicML as cml

@cml.benchmarks.memory_monitor
def create_model():
    """创建一个模型"""
    model = cml.models.BPNN(seed=2020)
    return model

if __name__ == '__main__':
    create_model()
```

终端的大概会显示类似如下的内容：

```python
INFO:classicML:正在使用 Python 引擎
INFO:classicML:占用内存 55.52734 MB
```

## average_timer

内存监视装饰器.

```python
@cml.benchmarks.average_timer(repeat=5)
```

### 参数

* <b>repeat</b>: 整数，重复运行的次数.

### 注意

* 使用该函数统计平均计时会明显降低运行速度，请在开发时使用，避免在训练模型时使用.

## memory_monitor

内存监视装饰器.

```python
@cml.benchmarks.memory_monitor
```

### 注意

* 使用该函数统计内存信息，有潜在降低运行速度的可能性. 并且psutil针对的```Python```优化手段会导致在```CC```引擎的速度大幅降低.

## timer

程序计时装饰器.

```python
@cml.benchmarks.timer
```

