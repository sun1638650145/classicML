//
// ops.h
// ops
//
// Created by 孙瑞琦 on 2020/10/10.
// Refactor by 孙瑞琦 on 2021/12/29.
//
//

#ifndef CLASSICML_BACKEND_CC_OPS_OPS_H_
#define CLASSICML_BACKEND_CC_OPS_OPS_H_

// 使得在win64平台下可以正常展开M_PI.
#ifdef _WIN64
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "pybind11/pybind11.h"

#include "../dtypes.h"
#include "../matrix_op.h"

namespace ops {
// overload匹配顺序: 优先匹配存在y的情况.
template<typename XMatrix, typename YMatrix>
std::tuple<XMatrix, YMatrix> BootstrapSampling1(const XMatrix &x, const YMatrix &y, std::optional<uint32> seed);
template<typename Matrix>
Matrix BootstrapSampling2(const Matrix &x, const pybind11::object &y, std::optional<uint32> seed); // 用于匹配y为None的情况.

// 使用Graham扫描算法计算二维凸包.
// 由于仅在绘图中使用, 故单精度提高绘制速度.
class ConvexHull {
    public:
        explicit ConvexHull(matrix32 points);
        virtual ~ConvexHull() = default;

        matrix32 ComputeConvexHull();

    public:
        matrix32 points;
        matrix32 hull;

    private:
        static void MatrixSorting(matrix32 &mat);
        static float32 GetSlope(const matrix32 &p0, const matrix32 &p1);
        static float32 GetCrossProduct(const matrix32 &p0, const matrix32 &p1, const matrix32 &p2);
};

template<typename Matrix>
Matrix CalculateCentroids(const Matrix &x, const row_vector32i &clusters);

template<typename Matrix, typename Vector, typename Array>
Matrix CalculateError(const Matrix &x,
                      const Vector &y,
                      const uint32 &i,
                      const pybind11::object &kernel,
                      const Vector &alphas,
                      const Vector &non_zero_alphas,
                      const Matrix &b);

template<typename Matrix>
Matrix CalculateEuclideanDistance(const Matrix &x0, const Matrix &x1);

// TODO(Steve Sun, tag:code):
//  overload匹配顺序: 经软件测试发现只使用单个模板参数时, np类型被pure类型兼容处理, 因此使用返回值模板参数和参数模板参数.
template<typename RFloat, typename PFloat>
RFloat ClipAlpha(PFloat &alpha, PFloat &low, PFloat &high);

template<typename Matrix, typename Float>
row_vector_bool CompareDifferences(const Matrix &x0, const Matrix &x1, Float &tol);

template<typename Matrix>
row_vector32i GetCluster(const Matrix &distances);

template<typename Float, typename Uint>
Float GetConditionalProbability(Uint &samples_on_attribute,
                                Uint &samples_in_category,
                                Uint &num_of_categories,
                                const bool &smoothing);

template<typename Float, typename Uint>
Float GetDependentPriorProbability(Uint &samples_on_attribute_in_category,
                                   Uint &number_of_sample,
                                   Uint &values_on_attribute,
                                   const bool &smoothing);

template<typename Dtype, typename RowVector>
std::tuple<Dtype, Dtype> GetPriorProbability(const uint32 &number_of_sample,
                                             const RowVector &y,
                                             const bool &smoothing);

template<typename RFloat, typename PFloat>
RFloat GetProbabilityDensity(PFloat &sample, PFloat &mean, PFloat &var);

// DEPRECATED(Steve R. Sun): `ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.
matrix64 GetW(const matrix64 &S_w, const matrix64 &mu_0, const matrix64 &mu_1);

template<typename Matrix>
Matrix GetW_V2(const Matrix &S_w, const Matrix &mu_0, const Matrix &mu_1);

template<typename Matrix>
Matrix GetWithinClassScatterMatrix(const Matrix &X_0, const Matrix &X_1, const Matrix &mu_0, const Matrix &mu_1);

template<typename Matrix, typename Dtype>
Matrix InitCentroids1(const Matrix &x, const Dtype &n_clusters, const std::string &init);
template<typename Matrix, typename Dtype>
Matrix InitCentroids2(const Matrix &x, const Dtype &n_clusters, Matrix init);

template<typename Dtype, typename RowVector>
std::tuple<uint32, Dtype> SelectSecondAlpha(Dtype &error,
                                            const RowVector &error_cache,
                                            const RowVector &non_bound_alphas);

// DEPRECATED(Steve R. Sun): `ops.cc_type_of_target`已经被弃用, 它将在未来的正式版本中被移除, 请使用`ops.cc_type_of_target_v2`.
std::string TypeOfTarget(const matrix64 &y);
std::string TypeOfTarget(const matrix64i &y);
std::string TypeOfTarget(const pybind11::array &y);

std::string TypeOfTarget_V2(const pybind11::array &y);
} // namespace ops

#endif /* CLASSICML_BACKEND_CC_OPS_OPS_H_ */