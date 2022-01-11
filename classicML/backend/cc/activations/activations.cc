//
// activations.cc
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#include "activations.h"

namespace activations {
Activation::Activation() {
    this->name = "activation";
}

Activation::Activation(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix> Matrix Activation::PyCall(const Matrix &z) {
    throw exceptions::NotImplementedError();
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix Activation::Diff(const Matrix &output,
                        const Matrix &a,
                        const pybind11::args &args,
                        const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}

Relu::Relu() {
    this->name = "relu";
}

Relu::Relu(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix> Matrix Relu::PyCall(const Matrix &z) {
    Matrix result(z.rows(), z.cols());

    for (int32 row = 0; row < z.rows(); row ++) {
        for (int32 col = 0; col < z.cols(); col ++) {
            if (0 >= z(row, col)) {
                result(row, col) = 0;
            } else {
                result(row, col) = z(row, col);
            }
        }
    }

    return result;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 计算函数的微分, 输入为前向传播输出的张量和输入的张量.
template<typename Matrix>
Matrix Relu::Diff(const Matrix &output,
                  const Matrix &a,
                  const pybind11::args &args,
                  const pybind11::kwargs &kwargs) {
    Matrix da = output;

    for (int32 row = 0; row < a.rows(); row ++) {
        for (int32 col = 0; col < a.cols(); col ++) {
            if (a(row, col) <= 0) {
                da(row, col) = 0;
            }
        }
    }

    return da;
}

Sigmoid::Sigmoid() {
    this->name = "sigmoid";
}

Sigmoid::Sigmoid(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix> Matrix Sigmoid::PyCall(const Matrix &z) {
    Matrix result(z.rows(), z.cols());
    result = 1 / (1 + (-z.array()).exp());

    return result;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 计算函数的微分, 输入为输出的张量, 输入的张量和真实的标签.
template<typename Matrix>
Matrix Sigmoid::Diff(const Matrix &output,
                     const Matrix &a,
                     const pybind11::args &args,
                     const pybind11::kwargs &kwargs) {
    Matrix y_true = pybind11::cast<Matrix>(args[0]);
    Matrix error = y_true - output;
    Matrix da = a.array() * (1 - a.array()) * error.array();

    return da;
}

Softmax::Softmax() {
    this->name = "softmax";
}

Softmax::Softmax(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix>
Matrix Softmax::PyCall(const Matrix &z) {
    Matrix temp_z = z;
    Matrix result = z;

    // 为了避免溢出减去最大值
    temp_z = temp_z.array() - z.maxCoeff();
    temp_z = temp_z.array().exp();

    for (int32 row = 0; row < z.rows(); row ++) {
        for (int32 col = 0; col < z.cols(); col ++) {
            result(row, col) = temp_z(row, col) / temp_z.row(row).sum();
        }
    }

    return result;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// Softmax函数的微分, 输入为输出的张量, 输入的张量和真实的标签.
template<typename Matrix>
Matrix Softmax::Diff(const Matrix &output,
                     const Matrix &a,
                     const pybind11::args &args,
                     const pybind11::kwargs &kwargs) {
    Matrix da = a - output;

    return da;
}

// 显式实例化.
template matrix32 Activation::PyCall(const matrix32 &z);
template matrix64 Activation::PyCall(const matrix64 &z);
template matrix32 Activation::Diff(const matrix32 &output,
                                   const matrix32 &a,
                                   const pybind11::args &args,
                                   const pybind11::kwargs &kwargs);
template matrix64 Activation::Diff(const matrix64 &output,
                                   const matrix64 &a,
                                   const pybind11::args &args,
                                   const pybind11::kwargs &kwargs);

template matrix32 Relu::PyCall(const matrix32 &z);
template matrix64 Relu::PyCall(const matrix64 &z);
template matrix32 Relu::Diff(const matrix32 &output,
                             const matrix32 &a,
                             const pybind11::args &args,
                             const pybind11::kwargs &kwargs);
template matrix64 Relu::Diff(const matrix64 &output,
                             const matrix64 &a,
                             const pybind11::args &args,
                             const pybind11::kwargs &kwargs);

template matrix32 Sigmoid::PyCall(const matrix32 &z);
template matrix64 Sigmoid::PyCall(const matrix64 &z);
template matrix32 Sigmoid::Diff(const matrix32 &output,
                                const matrix32 &a,
                                const pybind11::args &args,
                                const pybind11::kwargs &kwargs);
template matrix64 Sigmoid::Diff(const matrix64 &output,
                                const matrix64 &a,
                                const pybind11::args &args,
                                const pybind11::kwargs &kwargs);

template matrix32 Softmax::PyCall(const matrix32 &z);
template matrix64 Softmax::PyCall(const matrix64 &z);
template matrix32 Softmax::Diff(const matrix32 &output,
                                const matrix32 &a,
                                const pybind11::args &args,
                                const pybind11::kwargs &kwargs);
template matrix64 Softmax::Diff(const matrix64 &output,
                                const matrix64 &a,
                                const pybind11::args &args,
                                const pybind11::kwargs &kwargs);
}  // namespace activations