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
#include "pybind11/pybind11.h"

#include "dtypes.h"

namespace matrix_op {
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

Eigen::MatrixXd GetNonZeroSubMatrix(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &non_zero_mark);

std::vector<int32> NonZero(const Eigen::RowVectorXd &array);

template <typename Matrix, typename Dtype>
Matrix Reshape(Matrix matrix, const Dtype &row, const Dtype &column);
}  // namespace matrix_op

#endif /* CLASSICML_BACKEND_CC_MATRIX_OP_H_ */
