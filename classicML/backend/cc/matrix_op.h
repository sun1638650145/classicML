//
//  matrix_op.h
//  matrix_op
//
//  Created by 孙瑞琦 on 2020/12/8.
//

#ifndef MATRIX_OP_H
#define MATRIX_OP_H

#include <vector>

#include "Eigen/Dense"
#include "pybind11/pybind11.h"

namespace matrix_op {
Eigen::MatrixXd BroadcastSub(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

Eigen::MatrixXd GetNonZeroSubMatrix(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &non_zero_mark);

std::vector<int> NonZero(const Eigen::RowVectorXd &array);

Eigen::MatrixXd Reshape(Eigen::MatrixXd matrix, const int &row, const int &column);
}  // namespace matrix_op

#endif /* MATRIX_OP_H */
