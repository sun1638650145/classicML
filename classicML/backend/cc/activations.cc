//
// activations.cc
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#include "activations.h"

activations::Activation::Activation() {
    this->name = "activation";
}

activations::Activation::Activation(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix> Matrix activations::Activation::PyCall(const Matrix &z) {
    throw exceptions::NotImplementedError();
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix activations::Activation::Diff(const Matrix &output,
                                     const Matrix &a,
                                     const pybind11::args &args,
                                     const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}

activations::Relu::Relu() {
    this->name = "relu";
}

activations::Relu::Relu(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix> Matrix activations::Relu::PyCall(const Matrix &z) {
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
Matrix activations::Relu::Diff(const Matrix &output,
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

activations::Sigmoid::Sigmoid() {
    this->name = "sigmoid";
}

activations::Sigmoid::Sigmoid(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix> Matrix activations::Sigmoid::PyCall(const Matrix &z) {
    Matrix result(z.rows(), z.cols());
    result = 1 / (1 + (-z.array()).exp());

    return result;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 计算函数的微分, 输入为输出的张量, 输入的张量和真实的标签.
template<typename Matrix>
Matrix activations::Sigmoid::Diff(const Matrix &output,
                                  const Matrix &a,
                                  const pybind11::args &args,
                                  const pybind11::kwargs &kwargs) {
    Matrix y_true = pybind11::cast<Matrix>(args[0]);
    Matrix error = y_true - output;
    Matrix da = a.array() * (1 - a.array()) * error.array();

    return da;
}

activations::Softmax::Softmax() {
    this->name = "softmax";
}

activations::Softmax::Softmax(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 经过激活后的张量, 输入为张量.
template<typename Matrix>
Matrix activations::Softmax::PyCall(const Matrix &z) {
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
Matrix activations::Softmax::Diff(const Matrix &output,
                                  const Matrix &a,
                                  const pybind11::args &args,
                                  const pybind11::kwargs &kwargs) {
    Matrix da = a - output;

    return da;
}