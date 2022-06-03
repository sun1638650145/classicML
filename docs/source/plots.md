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

### plot_ensemble

```python
plot_ensemble(ensemble_model, x, y, x_label=None, y_label=None, plot_estimators=False)
```

可视化集成学习分类器二维示意图.

#### 参数

* <b>ensemble_model</b>: `classicML.models.ensemble`, 集成学习分类器实例.
* <b>x</b>: 一个 Numpy数组, 特征数据.
* <b>y</b>: 一个 Numpy数组, 标签.
* <b>x_label</b>: 字符串, 横轴的标签.
* <b>y_label</b>: 字符串, 纵轴的标签.
* <b>plot_estimators</b>: 布尔值, 是否绘制基学习器的分类边界.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### plot_history

```python
plot_history(history)
```

可视化历史记录.

#### 参数

* <b>history</b>: ```cml.backend.callbacks.History``` , callbacks实例.

### plot_k_means

```python
plot_k_means(k_means, x, x_label=None, y_label=None)
```

可视化K-均值聚类二维示意图.

#### 参数

* <b>lda</b>: ```cml.models.KMeans```, K-均值聚类实例.
* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>x_label</b>: 字符串，横轴的标签.
* <b>y_label</b>: 字符串，纵轴的标签.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### plot_linear_discriminant_analysis

```python
plot_linear_discriminant_analysis(lda, x, y, x_label=None, y_label=None)  # 可以使用缩写 plot_lda()
```

可视化线性判别分析二维示意图.

#### 参数

* <b>lda</b>: ```cml.models.LDA```，线性判别分析实例.
* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>x_label</b>: 字符串，横轴的标签.
* <b>y_label</b>: 字符串，纵轴的标签.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### plot_logistic_regression

```python
plot_logistic_regression(logistic_regression, x, y, x_label=None, y_label=None)
```

可视化逻辑回归二维示意图.

#### 参数

* <b>logistic_regression</b>: ```cml.models.LogisticRegression```，逻辑回归实例.
* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>x_label</b>: 字符串，横轴的标签.
* <b>y_label</b>: 字符串，纵轴的标签.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### plot_support_vector_classifier

```python
plot_support_vector_classifier(svc, x, y, x_label=None, y_label=None)  # 可以使用缩写 plot_svc()
```

可视化支持向量分类器二维示意图.

#### 参数

* <b>svc</b>: ```cml.models.SVC```，支持向量分类器实例.
* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>x_label</b>: 字符串，横轴的标签.
* <b>y_label</b>: 字符串，纵轴的标签.

#### 异常

* <b>ValueError</b>: 模型没有训练的错误.

### plot_tree

```python
plot_tree(tree)
```

绘制树的示意图.

#### 参数

* <b>tree</b>: ```cml.models.Tree```，树实例.

