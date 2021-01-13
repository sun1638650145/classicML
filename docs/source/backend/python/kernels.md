# kernels

classicML的核函数.

## kernel

核函数的基类.

```python
cml.kernels.Kernel(name=None)
```

### 参数

* <b>name</b>: 字符串，核函数名称.

### \__call__

函数实现.

```python
__call__(x_i, x_j)
```

#### 异常

* <b>NotImplementedError</b>: 函数没有实现.

