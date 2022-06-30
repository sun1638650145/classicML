# ops

classicML的底层核心操作.

## bootstrap_sampling

```python
bootstrap_sampling(x, y=None, seed=None)
```

对样本进行自助采样.

### 参数

* <b>x</b>: 一个 Numpy数组，特征数据.
* <b>y</b>: 一个 Numpy数组，数据样本(标签).
* <b>seed</b>: 整数，随机种子.

### 返回

自助采样后的新样本.

## ConvexHull

使用Graham扫描算法计算二维凸包.

```python
ConvexHull(points)
```

### 参数

* <b>points</b>: 一个Numpy数组或列表, 计算凸包的点.
* <b>hull</b>: 一个Numpy数组或列表, 凸包的点.

###  参考文献

* [Graham Scan Algorithm](https://lvngd.com/blog/convex-hull-graham-scan-algorithm-python/)

### compute_convex_hull

计算二维凸包.

```python
compute_convex_hull()
```

#### 返回

二维凸包.

## calculate_centroids

```python
calculate_centroids(x, clusters)
```

计算均值向量.

### 参数

* <b>x</b>: 一个 Numpy数组, 特征数据.
* <b>clusters</b>: 一个 Numpy数组, 当前的簇标记.

### 返回

均值向量.

## calculate_error

```python
calculate_error(x, y, i, kernel, alphas, non_zero_alphas, b)
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

## calculate_euclidean_distance

```python
calculate_euclidean_distance(x0, x1)
```

计算欧式距离.

### 参数

* <b>x0, x1</b>: Numpy数组, 要计算欧式距离的两个值.

### 返回

欧式距离.

## clip_alpha 

```python
clip_alpha(alpha, low, high)
```

修剪拉格朗日乘子.

### 参数

* <b>alpha</b>: 浮点数，拉格朗日乘子.
* <b>low</b>: 浮点数，正则化系数的下界.
* <b>high</b>: 浮点数，正则化系数的上界.

### 返回

修剪后的拉格朗日乘子.

## compare_differences 

```python
compare_differences(x0, x1, tol)
```

比较差异.

### 参数

* <b>x0, x1</b>: Numpy数组, 要比较差异的两个值.
* <b>tol</b>: 浮点数, 最小差异阈值.

### 返回

差异向量.

## get_cluster

```python
get_cluster(distances)
```

获取类条件概率.

### 参数

* <b>distances</b>: 一个Numpy, 距离.

### 返回

簇标记.

## get_conditional_probability

```python
get_conditional_probability(samples_on_attribute,
                            samples_in_category,
                            num_of_categories,
                            smoothing)
```

获取类条件概率.

### 参数

* <b>samples_on_attribute</b>: 整数，在某个属性的样本.
* <b>samples_in_category</b>: 整数，在某个类别上的样本.
* <b>num_of_categories</b>: 整数，类别的数量.
* <b>smoothing</b>: 布尔值，是否使用平滑.

### 返回

类条件概率.

## get_dependent_prior_probability

```python
get_dependent_prior_probability(samples_on_attribute_in_category,
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

## get_prior_probability

```python
get_prior_probability(number_of_sample, y, smoothing)
```

获取类先验概率.

### 参数

* <b>number_of_sample</b>: 整数，样本的总数.
* <b>y</b>: 一个 Numpy数组，标签.
* <b>smoothing</b>: 布尔值, 是否使用平滑.

### 返回

类先验概率.

## get_probability_density

```python
get_probability_density(sample, mean, var)
```

获得概率密度.

### 参数

* <b>sample</b>: 浮点数，样本的取值.
* <b>mean</b>: 浮点数，样本在某个属性的上的均值.
* <b>var</b>: 浮点数，样本在某个属性上的方差.

### 返回

概率密度.

## <del>get_w</del>

```python
get_w(S_w, mu_0, mu_1)
```

`get_w`已经被弃用, 它将在未来的正式版本中被移除, 请使用 `get_w_v2`.

获得投影向量.

### 参数

* <b>S_w</b>: 一个 Numpy数组，类内散度矩阵.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

投影向量.

## get_w_v2

```python
get_w_v2(S_w, mu_0, mu_1)
```

获得投影向量.

### 参数

* <b>S_w</b>: 一个 Numpy数组，类内散度矩阵.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

投影向量.

## get_within_class_scatter_matrix

```python
get_within_class_scatter_matrix(X_0, X_1, mu_0, mu_1)
```

获得类内散度矩阵.

### 参数

* <b>X_0</b>: 一个 Numpy数组，反例集合.
* <b>X_1</b>: 一个 Numpy数组，正例集合.
* <b>mu_0</b>: 一个 Numpy数组，反例的均值向量.
* <b>mu_1</b>: 一个 Numpy数组，正例的均值向量.

### 返回

类内散度矩阵.

## init_centroids

```python
init_centroids(x, n_clusters, init)
```

初始化初始均值向量.

### 参数

* <b>x</b>: 一个 Numpy数组，KKT条件的违背值缓存.
* <b>n_clusters</b>: 整数, 聚类簇的数量.
* <b>init</b>: 均值向量的初始化方式,
    * 'random': 采用随机初始化;
    * 列表或一个Numpy数组: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

### 返回

均值向量.

### 异常

* <b>ValueError</b>: 聚类簇数量与初始化均值向量数量不一致, 非法索引或不能自动转换的非法均值向量. 
* <b>TypeError</b>: 非法均值向量.

## select_second_alpha

```python
select_second_alpha(error, error_cache, non_bound_alphas)
```

选择第二个拉格朗日乘子，SMO采用的是启发式寻找的思想，找到目标函数变化量足够大，即选取变量样本间隔最大.

### 参数

* <b>error</b>: 浮点数，KKT条件的违背值.
* <b>error_cache</b>: 一个 Numpy数组，KKT条件的违背值缓存.
* <b>non_bound_alphas</b>: 一个 Numpy数组，非边界拉格朗日乘子.

### 返回

拉格朗日乘子的下标和违背值.

### 注意

* ```Python```和```CC```存在精度差异，存在潜在的可能性导致使用同样的数据和随机种子但不同后端的结果不一致，不过随着训练的轮数的增加，这种差异会逐渐消失.

## type_of_target

```python
type_of_target(y)
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

