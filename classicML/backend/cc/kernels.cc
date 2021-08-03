//
// kernels.cc
// kernels
//
// Create by 孙瑞琦 on 2021/2/10.
//
//

#include "kernels.h"

kernels::Kernel::Kernel() {
    this->name = "kernel";
}

kernels::Kernel::Kernel(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix kernels::Kernel::PyCall(const Matrix &x_i, const Matrix &x_j) {
    throw exceptions::NotImplementedError();  // 与Py后端实现相同, 主动抛出异常.
}

kernels::Linear::Linear() {
    this->name = "linear";
}

kernels::Linear::Linear(std::string name) {
    this->name = std::move(name);
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix>
Matrix kernels::Linear::PyCall(const Matrix &x_i, const Matrix &x_j) {
    // 预处理x_i, 避免一维张量无法处理.
    Matrix _x_i = x_i;
    if (x_i.cols() == 1) {
        _x_i = matrix_op::Reshape(x_i, 1, -1);
    }

    Matrix kappa = x_j * _x_i.transpose();

    return matrix_op::Reshape(kappa, 1, -1);
}

kernels::Polynomial::Polynomial() {
    this->name = "poly";
    this->gamma = 1.0;
    this->degree = 3;
}

kernels::Polynomial::Polynomial(std::string name, float64 gamma, int32 degree) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->degree = degree;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix>
Matrix kernels::Polynomial::PyCall(const Matrix &x_i, const Matrix &x_j) {
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

kernels::RBF::RBF() {
    this->name = "rbf";
    this->gamma = 1.0;
}

kernels::RBF::RBF(std::string name, float64 gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix> Matrix kernels::RBF::PyCall(const Matrix &x_i, const Matrix &x_j) {
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

kernels::Gaussian::Gaussian() {
    this->name = "gaussian";
    this->gamma = 1.0;
}

kernels::Gaussian::Gaussian(std::string name, float64 gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

kernels::Sigmoid::Sigmoid() {
    this->name = "sigmoid";
    this->gamma = 1.0;
    this->beta = 1.0;
    this->theta = -1.0;
}

kernels::Sigmoid::Sigmoid(std::string name, float64 gamma, float64 beta, float64 theta) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->beta = beta;
    this->theta = theta;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
template<typename Matrix> Matrix kernels::Sigmoid::PyCall(const Matrix &x_i, const Matrix &x_j) {
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