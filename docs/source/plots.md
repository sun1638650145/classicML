# plots

classicML 提供了一系列的工具帮助你更好的理解和分析你设计的模型，其中```cml.plots``` API就是用来可视化模型的辅助工具，承接之前的例子支持向量机的例子，你仅仅需要增加一行代码就可以可视化支持向量机

```python
import classicML as cml
# 实例化一个支持向量机
model = cml.models.SVC(seed=2020)
# 编译模型参数，配置软间隔系数和核函数
model.compile(C=10000.0, kernel='rbf')
# 训练模型
model.fit(x, y)
# 在测试集上测试
y_pred = model.predict(x_test)
# 可视化模型
cml.plots.plot_svc(model, x, y, '密度', '含糖率')
```

## 可视化工具

### plot_bayes

```python
cml.plots.plot_bayes(bayes, x, y)
```

 可视化朴素贝叶斯分类器或超父独依赖估计器的二维示意图.

#### 参数

* <b>bayes</b>: ```cml.models.NB``` 或者```cml.models.SPODE```，朴素贝叶斯分类器或超父独依赖估计器实例.
* <b>x</b>: 一个 Numpy数组，或者是Pandas的DataFrame，特征数据.
* <b>y</b>: 一个 Numpy数组，或者是Pandas的DataFrame，标签.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.