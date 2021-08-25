//
//  matrix_op.h
//  matrix_op
//
//  Created by 孙瑞琦 on 2020/12/8.
//

#ifndef CLASSICML_BACKEND_CC_MATRIX_OP_H_
#define CLASSICML_BACKEND_CC_MATRIX_OP_H_

#include <ctime>
#include <random>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dtypes.h"

namespace matrix_op {
bool AnyDiscreteInteger(const pybind11::array &array);

template <typename Matrix>
Matrix BroadcastSub(const Matrix &a, const Matrix &b);

template<typename Matrix, typename Dtype>
Matrix GenerateRandomStandardNormalDistributionMatrix(const int32 &rows,
                                                      const int32 &columns,
                                                      const std::optional<uint32> &seed);

template<typename Matrix, typename Dtype>
Matrix GenerateRandomUniformDistributionMatrix(const int32 &rows,
                                               const int32 &columns,
                                               const std::optional<uint32> &seed);

template<typename Matrix, typename Vector>
Matrix GetNonZeroSubMatrix(const Matrix &matrix, const Vector &non_zero_mark);

template<typename RowVector>
std::vector<uint32> NonZero(const RowVector &array);

template <typename Matrix, typename Dtype>
Matrix Reshape(Matrix matrix, const Dtype &row, const Dtype &column);

std::variant<std::set<float32>, std::set<uint8>> Unique(const pybind11::array &array);
}  // namespace matrix_op

#endif /* CLASSICML_BACKEND_CC_MATRIX_OP_H_ */
