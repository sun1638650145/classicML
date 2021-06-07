//
//  ops.h
//  ops
//
//  Created by 孙瑞琦 on 2020/10/10.
//

#ifndef CLASSICML_BACKEND_CC_OPS_H_
#define CLASSICML_BACKEND_CC_OPS_H_

#include "Eigen/Core"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "matrix_op.h"

namespace ops {
Eigen::MatrixXd CalculateError(const Eigen::MatrixXd &x,
                               const Eigen::MatrixXd &y,
                               const int &i,
                               const pybind11::object &kernel,
                               const Eigen::MatrixXd &alphas,
                               const Eigen::VectorXd &non_zero_mark,
                               const Eigen::MatrixXd &b);

Eigen::ArrayXd ClipAlpha(const double &alpha, const double &low, const double &high);

double GetConditionalProbability(const double &samples_on_attribute,
                                 const int &samples_in_category,
                                 const int &num_of_categories,
                                 const bool &smoothing);

double GetDependentPriorProbability(const int &samples_on_attribute_in_category,
                                    const int &number_of_sample,
                                    const int &values_on_attribute,
                                    const bool &smoothing);

std::tuple<double, double> GetPriorProbability(const int &number_of_sample,
                                               const Eigen::RowVectorXd &y,
                                               const bool &smoothing);

double GetProbabilityDensity(const double &sample,
                             const double &mean,
                             const double &var);

// DEPRECATED(Steve R. Sun): `ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.
Eigen::MatrixXd GetW(const Eigen::MatrixXd &S_w, const Eigen::MatrixXd &mu_0, const Eigen::MatrixXd &mu_1);

Eigen::MatrixXd GetW_V2(const Eigen::MatrixXd &S_w, const Eigen::MatrixXd &mu_0, const Eigen::MatrixXd &mu_1);

Eigen::MatrixXd GetWithinClassScatterMatrix(const Eigen::MatrixXd &X_0,
                                            const Eigen::MatrixXd &X_1,
                                            const Eigen::MatrixXd &mu_0,
                                            const Eigen::MatrixXd &mu_1);

std::tuple<int, double> SelectSecondAlpha(const double &error,
                                          const Eigen::RowVectorXd &error_cache,
                                          const Eigen::RowVectorXd &non_bound_alphas);

// overload
std::string TypeOfTarget(const Eigen::MatrixXd &y);
std::string TypeOfTarget(const Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> &y);
std::string TypeOfTarget(const pybind11::array &y);
}  // namespace ops

PYBIND11_MODULE(ops, m) {
    m.doc() = R"pbdoc(classicML的底层核心操作, 以C++实现)pbdoc";

    m.def("cc_calculate_error", &ops::CalculateError, R"pbdoc(
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
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("i"),
          pybind11::arg("kernel"), pybind11::arg("alphas"), pybind11::arg("non_zero_alphas"),
          pybind11::arg("b"));

    m.def("cc_clip_alpha", &ops::ClipAlpha, R"pbdoc(
修剪拉格朗日乘子.

    Arguments:
        alpha: numpy.ndarray, 拉格朗日乘子.
        low: float, 正则化系数的下界.
        high: float, 正则化系数的上界.

    Returns:
        修剪后的拉格朗日乘子.)pbdoc",
          pybind11::arg("alpha"), pybind11::arg("low"), pybind11::arg("high"));

    m.def("cc_get_conditional_probability", &ops::GetConditionalProbability, R"pbdoc(
获取类条件概率.

    Arguments:
        samples_on_attribute: float, 在某个属性的样本.
        samples_in_category: float, 在某个类别上的样本.
        num_of_categories: int, 类别的数量.
        smoothing: bool, 是否使用平滑.

    Returns:
        类条件概率.)pbdoc",
          pybind11::arg("samples_on_attribute"), pybind11::arg("samples_in_category"),
          pybind11::arg("num_of_categories"), pybind11::arg("smoothing"));

    m.def("cc_get_dependent_prior_probability", &ops::GetDependentPriorProbability, R"pbdoc(
获取有依赖的类先验概率.

    Arguments:
        samples_on_attribute_in_category: int, 类别为c的属性i上取值为xi的样本.
        number_of_sample: int, 样本的总数.
        values_on_attribute: int, 在属性i上的取值数.
        smoothing: bool, 是否使用平滑.

    Returns:
        先验概率.)pbdoc",
          pybind11::arg("samples_on_attribute_in_category"), pybind11::arg("number_of_sample"),
          pybind11::arg("values_on_attribute"), pybind11::arg("smoothing"));

    m.def("cc_get_prior_probability", &ops::GetPriorProbability, R"pbdoc(
获取类先验概率.

    Arguments:
        number_of_sample: int, 样本的总数.
        y: numpy.ndarray, 标签.
        smoothing: bool, 是否使用平滑.

    Returns:
        类先验概率.)pbdoc",
          pybind11::arg("number_of_sample"), pybind11::arg("y"), pybind11::arg("smoothing"));

    m.def("cc_get_probability_density", &ops::GetProbabilityDensity, R"pbdoc(
获得概率密度.

    Arguments:
        sample: float, 样本的取值.
        mean: float, 样本在某个属性的上的均值.
        var: float, 样本在某个属性上的方差.

    Returns:
        概率密度.)pbdoc",
          pybind11::arg("sample"), pybind11::arg("mean"), pybind11::arg("var"));

    m.def("cc_get_w", &ops::GetW, R"pbdoc(
获得投影向量.

    DEPRECATED:
        `ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.)pbdoc",
          pybind11::arg("S_w"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    m.def("cc_get_w_v2", &ops::GetW_V2, R"pbdoc(
获得投影向量.

    Arguments:
        S_w: numpy.ndarray, 类内散度矩阵.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        投影向量.)pbdoc",
          pybind11::arg("S_w"), pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    m.def("cc_get_within_class_scatter_matrix", &ops::GetWithinClassScatterMatrix, R"pbdoc(
获得类内散度矩阵.

    Arguments:
        X_0: numpy.ndarray, 反例集合.
        X_1: numpy.ndarray, 正例集合.
        mu_0: numpy.ndarray, 反例的均值向量.
        mu_1: numpy.ndarray, 正例的均值向量.

    Returns:
        类内散度矩阵.)pbdoc",
          pybind11::arg("X_0"), pybind11::arg("X_1"),
          pybind11::arg("mu_0"), pybind11::arg("mu_1"));

    m.def("cc_select_second_alpha", &ops::SelectSecondAlpha, R"pbdoc(
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
        拉格朗日乘子的下标和违背值.)pbdoc",
          pybind11::arg("error"), pybind11::arg("error_cache"),
          pybind11::arg("non_bound_alphas"));

    m.def("cc_type_of_target", pybind11::overload_cast<const Eigen::MatrixXd &>(&ops::TypeOfTarget), R"pbdoc(
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
        - 注意此函数为CC版本, 暂不能处理str类型的数据.)pbdoc",
          pybind11::arg("y"));
    m.def("cc_type_of_target",
          pybind11::overload_cast<const Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> &>(&ops::TypeOfTarget), R"pbdoc(
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
        - 注意此函数为CC版本, 暂不能处理str类型的数据.)pbdoc",
          pybind11::arg("y"));
    m.def("cc_type_of_target", pybind11::overload_cast<const pybind11::array &>(&ops::TypeOfTarget), R"pbdoc(
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
        - 注意此函数为CC版本, 暂不能处理str类型的数据.)pbdoc",
          pybind11::arg("y"));

    m.attr("__version__") = "backend.cc.ops.0.10.2";
}

#endif /* CLASSICML_BACKEND_CC_OPS_H_ */