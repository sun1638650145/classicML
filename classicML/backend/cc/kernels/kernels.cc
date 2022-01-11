//
// kernels.cc
// kernels
//
// Create by 孙瑞琦 on 2021/2/10.
//
//

#include "kernels.h"

namespace kernels {
Kernel::Kernel() {
    this->name = "kernel";
}

Kernel::Kernel(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix Kernel::PyCall(const Matrix &x_i, const Matrix &x_j) {
    throw exceptions::NotImplementedError();  // 与Py后端实现相同, 主动抛出异常.
}

Linear::Linear() {
    this->name = "linear";
}

Linear::Linear(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix>
Matrix Linear::PyCall(const Matrix &x_i, const Matrix &x_j) {
    // 预处理x_i, 避免一维张量无法处理.
    Matrix _x_i = x_i;
    if (x_i.cols() == 1) {
        _x_i = matrix_op::Reshape(x_i, 1, -1);
    }

    Matrix kappa = x_j * _x_i.transpose();

    return matrix_op::Reshape(kappa, 1, -1);
}

Polynomial::Polynomial() {
    this->name = "poly";
    this->gamma = 1.0;
    this->degree = 3;
}

Polynomial::Polynomial(std::string name, float64 gamma, int32 degree) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->degree = degree;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix>
Matrix Polynomial::PyCall(const Matrix &x_i, const Matrix &x_j) {
    // 预处理x_i, 避免一维张量无法处理.
    Matrix _x_i = x_i;
    if (x_i.cols() == 1) {
        _x_i = matrix_op::Reshape(x_i, 1, -1);
    }

    Matrix kappa = x_j * _x_i.transpose();
    kappa = kappa.array().pow(this->degree);
    kappa = this->gamma * kappa;

    return matrix_op::Reshape(kappa, 1, -1);
}

RBF::RBF() {
    this->name = "rbf";
    this->gamma = 1.0;
}

RBF::RBF(std::string name, float64 gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix> Matrix RBF::PyCall(const Matrix &x_i, const Matrix &x_j) {
    // 预处理x_i, 避免一维张量无法处理.
    Matrix _x_i = x_i;
    if (x_i.cols() == 1) {
        _x_i = matrix_op::Reshape(x_i, 1, -1);
    }

    Matrix temp = matrix_op::BroadcastSub(x_j, _x_i).array().pow(2);
    Matrix kappa = -temp.rowwise().sum();
    kappa = (this->gamma * kappa).array().exp();

    return matrix_op::Reshape(kappa, 1, -1);
}

Gaussian::Gaussian() {
    this->name = "gaussian";
    this->gamma = 1.0;
}

Gaussian::Gaussian(std::string name, float64 gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

Sigmoid::Sigmoid() {
    this->name = "sigmoid";
    this->gamma = 1.0;
    this->beta = 1.0;
    this->theta = -1.0;
}

Sigmoid::Sigmoid(std::string name, float64 gamma, float64 beta, float64 theta) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->beta = beta;
    this->theta = theta;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix> Matrix Sigmoid::PyCall(const Matrix &x_i, const Matrix &x_j) {
    // 预处理x_i, 避免一维张量无法处理.
    Matrix _x_i = x_i;
    if (x_i.cols() == 1) {
        _x_i = matrix_op::Reshape(x_i, 1, -1);
    }

    Matrix kappa = this->beta * (x_j * _x_i.transpose()).array() + this->theta;
    kappa = kappa.array().tanh();
    kappa = this->gamma * kappa;

    return matrix_op::Reshape(kappa, 1, -1);
}

// 显式实例化.
template matrix32 Kernel::PyCall(const matrix32 &x_i, const matrix32 &x_j);
template matrix64 Kernel::PyCall(const matrix64 &x_i, const matrix64 &x_j);

template matrix32 Linear::PyCall(const matrix32 &x_i, const matrix32 &x_j);
template matrix64 Linear::PyCall(const matrix64 &x_i, const matrix64 &x_j);

template matrix32 Polynomial::PyCall(const matrix32 &x_i, const matrix32 &x_j);
template matrix64 Polynomial::PyCall(const matrix64 &x_i, const matrix64 &x_j);

template matrix32 RBF::PyCall(const matrix32 &x_i, const matrix32 &x_j);
template matrix64 RBF::PyCall(const matrix64 &x_i, const matrix64 &x_j);

template matrix32 Sigmoid::PyCall(const matrix32 &x_i, const matrix32 &x_j);
template matrix64 Sigmoid::PyCall(const matrix64 &x_i, const matrix64 &x_j);
}  // namespace kernels