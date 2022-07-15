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
//  由于代码共用的问题, _utils/callbacks模块尽管不需要matrix类型, 但是还是需要在编译时添加路径.
#include "Eigen/Core"
#include "pybind11/numpy.h"

#include "numpy_patch.h"

typedef float float32;
typedef double float64;

typedef std::int32_t int32;
typedef std::int64_t int64;

typedef std::uint8_t uint8;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

typedef pybind11::numpy_scalar<float32> np_float32;
typedef pybind11::numpy_scalar<float64> np_float64;

typedef pybind11::numpy_scalar<int32> np_int32;
typedef pybind11::numpy_scalar<int64> np_int64;
typedef pybind11::numpy_scalar<uint32> np_uint32;
typedef pybind11::numpy_scalar<uint64> np_uint64;

typedef Eigen::Matrix<bool, -1, -1> matrix_bool;
typedef Eigen::MatrixXf matrix32;
typedef Eigen::MatrixXd matrix64;
typedef Eigen::MatrixXi matrix32i;
typedef Eigen::Matrix<int64, -1, -1> matrix64i;

typedef Eigen::VectorXf vector32;
typedef Eigen::VectorXd vector64;
typedef Eigen::VectorXi vector32i;

typedef Eigen::RowVectorXf row_vector32f;
typedef Eigen::RowVectorXd row_vector64f;
typedef Eigen::RowVectorXi row_vector32i;
typedef Eigen::Matrix<int64, 1, -1> row_vector64i;
typedef Eigen::Matrix<bool, 1, -1> row_vector_bool;

typedef Eigen::ArrayXf array32;
typedef Eigen::ArrayXd array64;

typedef pybind11::array_t<float32> tensor32;
typedef pybind11::array_t<float64> tensor64;

#endif /* CLASSICML_BACKEND_CC_DTYPES_H_ */