//
// dtypes.h
// dtypes
//
// Created by 孙瑞琦 on 2021/7/19.
//
//

#ifndef CLASSICML_BACKEND_CC_DTYPES_H_
#define CLASSICML_BACKEND_CC_DTYPES_H_

// TODO(Steve R. Sun, tag:code):
//  由于代码共用的问题, _utils/callbacks模块尽管不需要matrix类型, 但是还是需要在编译时添加Eigen路径.
#include "Eigen/Core"

#include "numpy_patch.h"

typedef float float32;
typedef double float64;

typedef std::int32_t int32;
typedef std::int64_t int64;

typedef std::uint8_t uint8;
typedef std::uint32_t uint32;

typedef pybind11::numpy_scalar<int32> np_int32;
typedef pybind11::numpy_scalar<int64> np_int64;

typedef Eigen::MatrixXf matrix32;
typedef Eigen::MatrixXd matrix64;

typedef Eigen::RowVectorXi row_vector32i;
typedef Eigen::Matrix<std::int64_t, 1, -1> row_vector64i;

#endif /* CLASSICML_BACKEND_CC_DTYPES_H_ */