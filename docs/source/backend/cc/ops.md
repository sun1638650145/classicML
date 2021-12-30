# ops

classicML的底层核心操作.

## cc_calculate_error

```python
cc_calculate_error(x, y, i, kernel, alphas, non_zero_alphas, b)
```

计算KKT条件的违背值.

### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>i</b>: 整数，第i个样本.
* <b>kernel</b>: ```cml.kernel.Kernels```实例，分类器使用的核函数.
* <b>alphas</b>: 一个 Numpy数组，拉格朗日乘子.
* <b>non_zero_alphas</b>: 一个 Numpy数组，非零拉格朗日乘子.
* <b>b</b>: 浮点数，偏置项.

### 返回

KKT条件的违背值.

## cc_clip_alpha 

```python
cc_clip_alpha(alpha, low, high)
```

修剪拉格朗日乘子.

### 参数

* <b>alpha</b>: 浮点数，拉格朗日乘子.
* <b>low</b>: 浮点数，正则化系数的下界.
* <b>high</b>: 浮点数，正则化系数的上界.

### 返回
修剪后的拉格朗日乘子.

## cc_get_conditional_probability

```python
cc_get_conditional_probability(samples_on_attribute, samples_in_category, num_of_categories, smoothing)
```

获取类条件概率.

### 参数

* <b>samples_on_attribute</b>: 整数，在某个属性的样本.
* <b>samples_in_category</b>: 整数，在某个类别上的样本.
* <b>num_of_categories</b>: 整数，类别的数量.
* <b>smoothing</b>: 布尔值，是否使用平滑.

### 返回

类条件概率.

## cc_get_dependent_prior_probability

```python
cc_get_dependent_prior_probability(samples_on_attribute_in_category,
                                   number_of_sample,
                                   values_on_attribute,
                                   smoothing)
```

获取有依赖的类先验概率.

### 参数

* <b>samples_on_attribute_in_category</b>: 整数，类别为c的属性i上取值为xi的样本.
* <b>number_of_sample</b>: 整数，样本的总数.
* <b>values_on_attribute</b>: 整数，在属性i上的取值数.
* <b>smoothing</b>: 布尔值, 是否使用平滑.

### 返回

类先验概率.

## cc_get_prior_probability

```python
cc_get_prior_probability(number_of_sample, y, smoothing)
```

获取类先验概率.

### 参数

* <b>number_of_sample</b>: 整数，样本的总数.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>smoothing</b>: 布尔值, 是否使用平滑.

### 返回

类先验概率.

## cc_get_probability_density

```python
cc_get_probability_density(sample, mean, var)
```

获得概率密度.

### 参数

* <b>sample</b>: 浮点数，样本的取值.
* <b>mean</b>: 浮点数，样本在某个属性的上的均值.
* <b>var</b>: 浮点数，样本在某个属性上的方差.

### 返回

概率密度.

## <del>cc_get_w</del>

```python
cc_get_w(S_w, mu_0, mu_1)
```

`cc_get_w`已经被弃用, 它将在未来的正式版本中被移除, 请使用 `cc_get_w_v2`.

获得投影向量.

### 参数

* <b>S_w</b>: 一个 Numpy数组，类内散度矩阵.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

投影向量.

## cc_get_w_v2

```python
cc_get_w_v2(S_w, mu_0, mu_1)
```

获得投影向量.

### 参数

* <b>S_w</b>: 一个 Numpy数组，类内散度矩阵.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

投影向量.

## cc_get_within_class_scatter_matrix

```python
cc_get_within_class_scatter_matrix(X_0, X_1, mu_0, mu_1)
```

获得类内散度矩阵.

### 参数

* <b>X_0</b>: 一个 Numpy数组，反例集合.
* <b>X_1</b>: 一个 Numpy数组，正例集合.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

类内散度矩阵.

## cc_select_second_alpha

```python
cc_select_second_alpha(error, error_cache, non_bound_alphas)
```

选择第二个拉格朗日乘子，SMO采用的是启发式寻找的思想，找到目标函数变化量足够大，即选取变量样本间隔最大.

### 参数

* <b>error</b>: 浮点数，KKT条件的违背值.
* <b>error_cache</b>: 一个 Numpy数组，KKT条件的违背值缓存.
* <b>non_bound_alphas</b>: 一个 Numpy数组，非边界拉格朗日乘子.

### 返回

拉格朗日乘子的下标和违背值.

## <del>cc_type_of_target</del>

```python
cc_type_of_target(y)
```

`cc_type_of_target`已经被弃用, 它将在未来的正式版本中被移除, 请使用 `cc_type_of_target_v2`.

判断输入数据的类型.

### 参数

* <b>y</b>: 一个 Numpy数组，待判断类型的数据.

### 返回

* ```'binary'```: 元素只有两个离散值，类型不限.
* ```'continuous'```: 元素都是浮点数，且不是对应整数的浮点数.
* ```'multiclass'```: 元素不只有两个离散值，类型不限.
* ```'multilabel'```: 元素标签不为一，类型不限.
* ```'unknown'```: 类型未知.

## cc_type_of_target_v2

```python
cc_type_of_target_v2(y)
```

判断输入数据的类型.

### 参数

* <b>y</b>: 一个 Numpy数组，待判断类型的数据.

### 返回

* ```'binary'```: 元素只有两个离散值，类型不限.
* ```'continuous'```: 元素都是浮点数，且不是对应整数的浮点数.
* ```'multiclass'```: 元素不只有两个离散值，类型不限.
* ```'multilabel'```: 元素标签不为一，类型不限.
* ```'unknown'```: 类型未知.

