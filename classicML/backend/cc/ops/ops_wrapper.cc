//
// ops_wrapper.cc
// ops
//
// Created by 孙瑞琦 on 2020/12/28.
//
//

#include "pybind11/pybind11.h"

#include "ops.h"

PYBIND11_MODULE(ops, m) {
    m.doc() = R"pbdoc(classicML的底层核心操作, 以C++实现)pbdoc";

    // Overloaded function.
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling1<matrix32, matrix32>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling1<matrix64, matrix64>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling1<matrix32, matrix32i>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling1<matrix64, matrix64i>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling2<matrix32>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());
    m.def("cc_bootstrap_sampling", &ops::BootstrapSampling2<matrix64>, R"pbdoc(
对样本进行自助采样.

    Args:
        x: numpy.ndarray, array-like, 数据样本.
        y: numpy.ndarray, array-like, default=None,
            数据样本(标签).
        seed: int, default=None,
            随机种子.

    Returns:
        自助采样后的新样本.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y")=pybind11::none(),
          pybind11::arg("seed")=pybind11::none());

    pybind11::class_<ops::ConvexHull>(m, "ConvexHull", R"pbdoc(
使用Graham扫描算法计算二维凸包.

    Attributes:
        points: np.ndarray or list, 计算凸包的点.
        hull: np.ndarray, 凸包的点.

    References:
        - [Graham Scan Algorithm](https://lvngd.com/blog/convex-hull-graham-scan-algorithm-python/)
)pbdoc")
        .def(pybind11::init<matrix32>(), R"pbdoc(
Args:
    points: np.ndarray or list, 计算凸包的点.
)pbdoc", pybind11::arg("points"))
        .def_readwrite("points", &ops::ConvexHull::points)
        .def_readonly("hull", &ops::ConvexHull::hull)
        .def("compute_convex_hull", &ops::ConvexHull::ComputeConvexHull, R"pbdoc(
计算二维凸包.

Return:
    二维凸包.
)pbdoc");

    // Overloaded function.
    m.def("cc_calculate_centroids", &ops::CalculateCentroids<matrix32>, R"pbdoc(
计算均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        clusters: numpy.ndarray, 当前的簇标记.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("clusters"));
    m.def("cc_calculate_centroids", &ops::CalculateCentroids<matrix64>, R"pbdoc(
计算均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        clusters: numpy.ndarray, 当前的簇标记.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("clusters"));

    // Overloaded function.
    m.def("cc_calculate_error", &ops::CalculateError<matrix32, vector32, array32>, R"pbdoc(
计算KKT条件的违背值.

    Arguments:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        i: int, 第i个样本.
        kernel: classicML.kernel.Kernels 实例, 分类器使用的核函数.
        alphas: numpy.ndarray, 拉格朗日乘子.
        non_zero_alphas: numpy.ndarray, 非零拉格朗日乘子.
        b: float, 偏置项.

    Returns:
        KKT条件的违背值.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y"),
          pybind11::arg("i"),
          pybind11::arg("kernel"),
          pybind11::arg("alphas"),
          pybind11::arg("non_zero_alphas"),
          pybind11::arg("b"));
    m.def("cc_calculate_error", &ops::CalculateError<matrix64, vector64, array64>, R"pbdoc(
计算KKT条件的违背值.

    Arguments:
        x: numpy.ndarray, array-like, 特征数据.
        y: numpy.ndarray, array-like, 标签.
        i: int, 第i个样本.
        kernel: classicML.kernel.Kernels 实例, 分类器使用的核函数.
        alphas: numpy.ndarray, 拉格朗日乘子.
        non_zero_alphas: numpy.ndarray, 非零拉格朗日乘子.
        b: float, 偏置项.

    Returns:
        KKT条件的违背值.)pbdoc",
          pybind11::arg("x"),
          pybind11::arg("y"),
          pybind11::arg("i"),
          pybind11::arg("kernel"),
          pybind11::arg("alphas"),
          pybind11::arg("non_zero_alphas"),
          pybind11::arg("b"));

    // Overloaded function.
    m.def("cc_calculate_euclidean_distance", &ops::CalculateEuclideanDistance<matrix32>, R"pbdoc(
计算欧式距离.

    Args:
        x0, x1: numpy.ndarray, 要计算欧式距离的两个值.

    Return:
        欧式距离.)pbdoc", pybind11::arg("x0"), pybind11::arg("x1"));
    m.def("cc_calculate_euclidean_distance", &ops::CalculateEuclideanDistance<matrix64>, R"pbdoc(
计算欧式距离.

    Args:
        x0, x1: numpy.ndarray, 要计算欧式距离的两个值.

    Return:
        欧式距离.)pbdoc", pybind11::arg("x0"), pybind11::arg("x1"));

    // Overloaded function.
    m.def("cc_clip_alpha", &ops::ClipAlpha<np_float32, np_float32>, R"pbdoc(
修剪拉格朗日乘子.

    Arguments:
        alpha: float, 拉格朗日乘子.
        low: float, 正则化系数的下界.
        high: float, 正则化系数的上界.

    Returns:
        修剪后的拉格朗日乘子.
)pbdoc", pybind11::arg("alpha"), pybind11::arg("low"), pybind11::arg("high"));
    m.def("cc_clip_alpha", &ops::ClipAlpha<np_float64, np_float64>, R"pbdoc(
修剪拉格朗日乘子.

    Arguments:
        alpha: float, 拉格朗日乘子.
        low: float, 正则化系数的下界.
        high: float, 正则化系数的上界.

    Returns:
        修剪后的拉格朗日乘子.
)pbdoc", pybind11::arg("alpha"), pybind11::arg("low"), pybind11::arg("high"));
    m.def("cc_clip_alpha", &ops::ClipAlpha<np_float32, float32>, R"pbdoc(
修剪拉格朗日乘子.

    Arguments:
        alpha: float, 拉格朗日乘子.
        low: float, 正则化系数的下界.
        high: float, 正则化系数的上界.

    Returns:
        修剪后的拉格朗日乘子.
)pbdoc", pybind11::arg("alpha"), pybind11::arg("low"), pybind11::arg("high"));

    // Overloaded function.
    m.def("cc_compare_differences", &ops::CompareDifferences<matrix32, np_float32>, R"pbdoc(
比较差异.

    Args:
        x0, x1: numpy.ndarray, 要比较差异的两个值.
        tol: float, 最小差异阈值.

    Return:
        差异向量.
)pbdoc", pybind11::arg("x0"), pybind11::arg("x1"), pybind11::arg("tol"));
    m.def("cc_compare_differences", &ops::CompareDifferences<matrix64, np_float64>, R"pbdoc(
比较差异.

    Args:
        x0, x1: numpy.ndarray, 要比较差异的两个值.
        tol: float, 最小差异阈值.

    Return:
        差异向量.
)pbdoc", pybind11::arg("x0"), pybind11::arg("x1"), pybind11::arg("tol"));
    m.def("cc_compare_differences", &ops::CompareDifferences<matrix32, float32>, R"pbdoc(
比较差异.

    Args:
        x0, x1: numpy.ndarray, 要比较差异的两个值.
        tol: float, 最小差异阈值.

    Return:
        差异向量.
)pbdoc", pybind11::arg("x0"), pybind11::arg("x1"), pybind11::arg("tol"));

    // Overloaded function.
    m.def("cc_get_cluster", &ops::GetCluster<matrix32>, R"pbdoc(
获得簇标记.

    Args:
        distances: numpy.ndarray, 距离.

    Return:
        簇标记.
)pbdoc", pybind11::arg("distances"));
    m.def("cc_get_cluster", &ops::GetCluster<matrix64>, R"pbdoc(
获得簇标记.

    Args:
        distances: numpy.ndarray, 距离.

    Return:
        簇标记.
)pbdoc", pybind11::arg("distances"));

    // Overloaded function.
    m.def("cc_get_conditional_probability", &ops::GetConditionalProbability<np_float32, np_uint32>, R"pbdoc(
获取类条件概率.

    Arguments:
        samples_on_attribute: int, 在某个属性的样本.
        samples_in_category: int, 在某个类别上的样本.
        num_of_categories: int, 类别的数量.
        smoothing: bool, 是否使用平滑.

    Returns:
        类条件概率.)pbdoc",
          pybind11::arg("samples_on_attribute"),
          pybind11::arg("samples_in_category"),
          pybind11::arg("num_of_categories"),
          pybind11::arg("smoothing"));
    m.def("cc_get_conditional_probability", &ops::GetConditionalProbability<np_float64, np_uint64>, R"pbdoc(
获取类条件概率.

    Arguments:
        samples_on_attribute: int, 在某个属性的样本.
        samples_in_category: int, 在某个类别上的样本.
        num_of_categories: int, 类别的数量.
        smoothing: bool, 是否使用平滑.

    Returns:
        类条件概率.)pbdoc",
          pybind11::arg("samples_on_attribute"),
          pybind11::arg("samples_in_category"),
          pybind11::arg("num_of_categories"),
          pybind11::arg("smoothing"));
    m.def("cc_get_conditional_probability", &ops::GetConditionalProbability<np_float32, uint32>, R"pbdoc(
获取类条件概率.

    Arguments:
        samples_on_attribute: int, 在某个属性的样本.
        samples_in_category: int, 在某个类别上的样本.
        num_of_categories: int, 类别的数量.
        smoothing: bool, 是否使用平滑.

    Returns:
        类条件概率.)pbdoc",
          pybind11::arg("samples_on_attribute"),
          pybind11::arg("samples_in_category"),
          pybind11::arg("num_of_categories"),
          pybind11::arg("smoothing"));

    // Overloaded function.
    m.def("cc_get_dependent_prior_probability", &ops::GetDependentPriorProbability<np_float32, np_uint32>, R"pbdoc(
获取有依赖的类先验概率.

    Arguments:
        samples_on_attribute_in_category: int, 类别为c的属性i上取值为xi的样本.
        number_of_sample: int, 样本的总数.
        values_on_attribute: int, 在属性i上的取值数.
        smoothing: bool, 是否使用平滑.

    Returns:
        先验概率.)pbdoc",
          pybind11::arg("samples_on_attribute_in_category"),
          pybind11::arg("number_of_sample"),
          pybind11::arg("values_on_attribute"),
          pybind11::arg("smoothing"));
    m.def("cc_get_dependent_prior_probability", &ops::GetDependentPriorProbability<np_float64, np_uint64>, R"pbdoc(
获取有依赖的类先验概率.

    Arguments:
        samples_on_attribute_in_category: int, 类别为c的属性i上取值为xi的样本.
        number_of_sample: int, 样本的总数.
        values_on_attribute: int, 在属性i上的取值数.
        smoothing: bool, 是否使用平滑.

    Returns:
        先验概率.)pbdoc",
          pybind11::arg("samples_on_attribute_in_category"),
          pybind11::arg("number_of_sample"),
          pybind11::arg("values_on_attribute"),
          pybind11::arg("smoothing"));
    m.def("cc_get_dependent_prior_probability", &ops::GetDependentPriorProbability<np_float32, uint32>, R"pbdoc(
获取有依赖的类先验概率.

    Arguments:
        samples_on_attribute_in_category: int, 类别为c的属性i上取值为xi的样本.
        number_of_sample: int, 样本的总数.
        values_on_attribute: int, 在属性i上的取值数.
        smoothing: bool, 是否使用平滑.

    Returns:
        先验概率.)pbdoc",
          pybind11::arg("samples_on_attribute_in_category"),
          pybind11::arg("number_of_sample"),
          pybind11::arg("values_on_attribute"),
          pybind11::arg("smoothing"));

    // Overloaded function.
    m.def("cc_get_prior_probability", &ops::GetPriorProbability<np_float32, row_vector32i>, R"pbdoc(
获取类先验概率.

    Arguments:
        number_of_sample: int, 样本的总数.
        y: numpy.ndarray, 标签.
        smoothing: bool, 是否使用平滑.

    Returns:
        类先验概率.
)pbdoc", pybind11::arg("number_of_sample"), pybind11::arg("y"), pybind11::arg("smoothing"));
    m.def("cc_get_prior_probability", &ops::GetPriorProbability<np_float64, row_vector64i>, R"pbdoc(
获取类先验概率.

    Arguments:
        number_of_sample: int, 样本的总数.
        y: numpy.ndarray, 标签.
        smoothing: bool, 是否使用平滑.

    Returns:
        类先验概率.
)pbdoc", pybind11::arg("number_of_sample"), pybind11::arg("y"), pybind11::arg("smoothing"));

    // Overloaded function.
    m.def("cc_get_probability_density", &ops::GetProbabilityDensity<np_float32, np_float32>, R"pbdoc(
获得概率密度.

    Arguments:
        sample: float, 样本的取值.
        mean: float, 样本在某个属性的上的均值.
        var: float, 样本在某个属性上的方差.

    Returns:
        概率密度.
)pbdoc", pybind11::arg("sample"), pybind11::arg("mean"), pybind11::arg("var"));
    m.def("cc_get_probability_density", &ops::GetProbabilityDensity<np_float64, np_float64>, R"pbdoc(
获得概率密度.

    Arguments:
        sample: float, 样本的取值.
        mean: float, 样本在某个属性的上的均值.
        var: float, 样本在某个属性上的方差.

    Returns:
        概率密度.
)pbdoc", pybind11::arg("sample"), pybind11::arg("mean"), pybind11::arg("var"));
    m.def("cc_get_probability_density", &ops::GetProbabilityDensity<np_float32, float32>, R"pbdoc(
获得概率密度.

    Arguments:
        sample: float, 样本的取值.
        mean: float, 样本在某个属性的上的均值.
        var: float, 样本在某个属性上的方差.

    Returns:
        概率密度.
)pbdoc", pybind11::arg("sample"), pybind11::arg("mean"), pybind11::arg("var"));

    m.def("cc_get_w", &ops::GetW, R"pbdoc(
获得投影向量.

    DEPRECATED:
        `ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.
)pbdoc", pybind11::arg("S_w"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    // Overloaded function.
    m.def("cc_get_w_v2", &ops::GetW_V2<matrix32>, R"pbdoc(
获得投影向量.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.
)pbdoc", pybind11::arg("S_w"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));
    m.def("cc_get_w_v2", &ops::GetW_V2<matrix64>, R"pbdoc(
获得投影向量.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.
)pbdoc", pybind11::arg("S_w"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    // Overloaded function.
    m.def("cc_get_within_class_scatter_matrix", &ops::GetWithinClassScatterMatrix<matrix32>, R"pbdoc(
获得类内散度矩阵.

    Arguments:
        X_0: numpy.ndarray, 反例集合.
        X_1: numpy.ndarray, 正例集合.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        类内散度矩阵.
)pbdoc", pybind11::arg("X_0"), pybind11::arg("X_1"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));
    m.def("cc_get_within_class_scatter_matrix", &ops::GetWithinClassScatterMatrix<matrix64>, R"pbdoc(
获得类内散度矩阵.

    Arguments:
        X_0: numpy.ndarray, 反例集合.
        X_1: numpy.ndarray, 正例集合.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        类内散度矩阵.
)pbdoc", pybind11::arg("X_0"), pybind11::arg("X_1"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    // Overloaded function.
    m.def("cc_init_centroids", &ops::InitCentroids1<matrix32, uint32>, R"pbdoc(
初始化初始均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        n_clusters: int, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("n_clusters"), pybind11::arg("init"));
    m.def("cc_init_centroids", &ops::InitCentroids1<matrix64, uint64>, R"pbdoc(
初始化初始均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        n_clusters: int, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("n_clusters"), pybind11::arg("init"));
    m.def("cc_init_centroids", &ops::InitCentroids2<matrix32, uint32>, R"pbdoc(
初始化初始均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        n_clusters: int, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("n_clusters"), pybind11::arg("init"));
    m.def("cc_init_centroids", &ops::InitCentroids2<matrix64, uint64>, R"pbdoc(
初始化初始均值向量.

    Args:
        x: numpy.ndarray, 特征数据.
        n_clusters: int, 聚类簇的数量.
        init: 'random', list or numpy.ndarray, 均值向量的初始化方式,
            'random': 采用随机初始化;
            list or numpy.ndarray: 可以指定训练数据的索引, 也可以直接给定具体的均值向量.

    Return:
        均值向量.)pbdoc", pybind11::arg("x"), pybind11::arg("n_clusters"), pybind11::arg("init"));

    // Overloaded function.
    m.def("cc_select_second_alpha", &ops::SelectSecondAlpha<np_float32, row_vector32f>, R"pbdoc(
选择第二个拉格朗日乘子, SMO采用的是启发式寻找的思想,
找到目标函数变化量足够大, 即选取变量样本间隔最大.

    Arguments:
        error: float,
            KKT条件的违背值.
        error_cache: numpy.ndarray,
            KKT条件的违背值缓存.
        non_bound_alphas: numpy.ndarray,
            非边界拉格朗日乘子.

    Returns:
        拉格朗日乘子的下标和违背值.
)pbdoc", pybind11::arg("error"), pybind11::arg("error_cache"), pybind11::arg("non_bound_alphas"));
    m.def("cc_select_second_alpha", &ops::SelectSecondAlpha<np_float64, row_vector64f>, R"pbdoc(
选择第二个拉格朗日乘子, SMO采用的是启发式寻找的思想,
找到目标函数变化量足够大, 即选取变量样本间隔最大.

    Arguments:
        error: float,
            KKT条件的违背值.
        error_cache: numpy.ndarray,
            KKT条件的违背值缓存.
        non_bound_alphas: numpy.ndarray,
            非边界拉格朗日乘子.

    Returns:
        拉格朗日乘子的下标和违背值.
)pbdoc", pybind11::arg("error"), pybind11::arg("error_cache"), pybind11::arg("non_bound_alphas"));
    m.def("cc_select_second_alpha", &ops::SelectSecondAlpha<float32, row_vector32f>, R"pbdoc(
选择第二个拉格朗日乘子, SMO采用的是启发式寻找的思想,
找到目标函数变化量足够大, 即选取变量样本间隔最大.

    Arguments:
        error: float,
            KKT条件的违背值.
        error_cache: numpy.ndarray,
            KKT条件的违背值缓存.
        non_bound_alphas: numpy.ndarray,
            非边界拉格朗日乘子.

    Returns:
        拉格朗日乘子的下标和违背值.
)pbdoc", pybind11::arg("error"), pybind11::arg("error_cache"), pybind11::arg("non_bound_alphas"));

    // Overloaded function.
    m.def("cc_type_of_target", pybind11::overload_cast<const matrix64 &>(&ops::TypeOfTarget), R"pbdoc(
判断输入数据的类型.

    DEPRECATED:
        `ops.cc_type_of_target` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.

    Arguments:
        y: numpy.ndarray,
            待判断类型的数据.

    Returns:
        'binary': 元素只有两个离散值, 类型不限.
        'continuous': 元素都是浮点数, 且不是对应整数的浮点数.
        'multiclass': 元素不只有两个离散值, 类型不限.
        'multilabel': 元素标签不为一, 类型不限.
        'unknown': 类型未知.

    Notes:
        - 注意此函数为CC版本, 暂不能处理str类型的数据.
)pbdoc", pybind11::arg("y"));
    m.def("cc_type_of_target", pybind11::overload_cast<const matrix64i &>(&ops::TypeOfTarget), R"pbdoc(
判断输入数据的类型.

    DEPRECATED:
        `ops.cc_type_of_target` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.

    Arguments:
        y: numpy.ndarray,
            待判断类型的数据.

    Returns:
        'binary': 元素只有两个离散值, 类型不限.
        'continuous': 元素都是浮点数, 且不是对应整数的浮点数.
        'multiclass': 元素不只有两个离散值, 类型不限.
        'multilabel': 元素标签不为一, 类型不限.
        'unknown': 类型未知.

    Notes:
        - 注意此函数为CC版本, 暂不能处理str类型的数据.
)pbdoc", pybind11::arg("y"));
    m.def("cc_type_of_target", pybind11::overload_cast<const pybind11::array &>(&ops::TypeOfTarget), R"pbdoc(
判断输入数据的类型.

    DEPRECATED:
        `ops.cc_type_of_target` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.

    Arguments:
        y: numpy.ndarray,
            待判断类型的数据.

    Returns:
        'binary': 元素只有两个离散值, 类型不限.
        'continuous': 元素都是浮点数, 且不是对应整数的浮点数.
        'multiclass': 元素不只有两个离散值, 类型不限.
        'multilabel': 元素标签不为一, 类型不限.
        'unknown': 类型未知.

    Notes:
        - 注意此函数为CC版本, 暂不能处理str类型的数据.
)pbdoc", pybind11::arg("y"));

    m.def("cc_type_of_target_v2", &ops::TypeOfTarget_V2, R"pbdoc(
判断输入数据的类型.

    Arguments:
        y: numpy.ndarray,
            待判断类型的数据.

    Returns:
        'binary': 元素只有两个离散值, 类型不限.
        'continuous': 元素都是浮点数, 且不是对应整数的浮点数.
        'multiclass': 元素不只有两个离散值, 类型不限.
        'multilabel': 元素标签不为一, 类型不限.
        'unknown': 类型未知.

    Notes:
        - 注意此函数为CC版本, 暂不能处理多字符的str类型的数据.
)pbdoc", pybind11::arg("y"));

    m.attr("__version__") = "backend.cc.ops.0.14b1";
}