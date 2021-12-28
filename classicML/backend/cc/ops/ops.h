//
//  ops.h
//  ops
//
//  Created by 孙瑞琦 on 2020/10/10.
//

#ifndef CLASSICML_BACKEND_CC_OPS_OPS_H_
#define CLASSICML_BACKEND_CC_OPS_OPS_H_

// 使得在win64平台下可以正常展开M_PI.
#ifdef _WIN64
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "pybind11/stl.h"

#include "../dtypes.h"
#include "../matrix_op.h"

namespace ops {
template<typename Matrix, typename Vector, typename Array>
Matrix CalculateError(const Matrix &x,
                      const Matrix &y,
                      const uint32 &i,
                      const pybind11::object &kernel,
                      const Matrix &alphas,
                      const Vector &non_zero_mark,
                      const Matrix &b);

// Overloaded function.
std::variant<Eigen::Array<float32, 1, 1>, Eigen::Array<float64, 1, 1>>
ClipAlpha(const pybind11::buffer &alpha, const pybind11::buffer &low, const pybind11::buffer &high);
Eigen::Array<float32, 1, 1> ClipAlpha(const float32 &alpha, const float32 &low, const float32 &high);

float64 GetConditionalProbability(const uint32 &samples_on_attribute,
                                  const uint32 &samples_in_category,
                                  const uint32 &num_of_categories,
                                  const bool &smoothing);

float64 GetDependentPriorProbability(const uint32 &samples_on_attribute_in_category,
                                     const uint32 &number_of_sample,
                                     const uint32 &values_on_attribute,
                                     const bool &smoothing);

template<typename Dtype, typename RowVector>
std::tuple<Dtype, Dtype> GetPriorProbability(const uint32 &number_of_sample,
                                             const RowVector &y,
                                             const bool &smoothing);

// Overloaded function.
std::variant<float32, float64> GetProbabilityDensity(const pybind11::buffer &sample,
                                                     const pybind11::buffer &mean,
                                                     const pybind11::buffer &var);
float32 GetProbabilityDensity(const float32 &sample, const float32 &mean, const float32 &var);

// DEPRECATED(Steve R. Sun): `ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.
matrix64 GetW(const matrix64 &S_w, const matrix64 &mu_0, const matrix64 &mu_1);

template <typename Matrix>
Matrix GetW_V2(const Matrix &S_w, const Matrix &mu_0, const Matrix &mu_1);

template <typename Matrix>
Matrix GetWithinClassScatterMatrix(const Matrix &X_0,
                                   const Matrix &X_1,
                                   const Matrix &mu_0,
                                   const Matrix &mu_1);

template<typename Dtype, typename RowVector>
std::tuple<uint32, Dtype> SelectSecondAlpha(const Dtype &error,
                                           const RowVector &error_cache,
                                           const RowVector &non_bound_alphas);

// DEPRECATED(Steve R. Sun): `ops.cc_type_of_target`已经被弃用, 它将在未来的正式版本中被移除, 请使用`ops.cc_type_of_target_v2`.
// Overloaded function.
std::string TypeOfTarget(const matrix64 &y);
std::string TypeOfTarget(const matrix64i &y);
std::string TypeOfTarget(const pybind11::array &y);

std::string TypeOfTarget_V2(const pybind11::array &y);
}  // namespace ops

#endif /* CLASSICML_BACKEND_CC_OPS_OPS_H_ */