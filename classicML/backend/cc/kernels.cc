//
// kernels.cc
// kernels
//
// Create by 孙瑞琦 on 2020/2/10.
//
//

#include "kernels.h"

Kernel::Kernel() {
    this->name = "kernel";
}

Kernel::Kernel(std::string name) {
    this->name = std::move(name);
}

Eigen::MatrixXd Kernel::PyCall(const Eigen::MatrixXd &x_i,
                               const Eigen::MatrixXd &x_j) {
    throw NotImplementedError();  // 与Py后端实现相同, 主动抛出异常.
}

Linear::Linear() {
    this->name = "linear";
}

Linear::Linear(std::string name) {
    this->name = std::move(name);
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd Linear::PyCall(const Eigen::MatrixXd &x_i,
                               const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = x_j * x_i.transpose();

    return kappa;
}

Polynomial::Polynomial() {
    this->name = "poly";
    this->gamma = 1.0;
    this->degree = 3;
}

Polynomial::Polynomial(std::string name, double gamma, int degree) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->degree = degree;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd Polynomial::PyCall(const Eigen::MatrixXd &x_i,
                                   const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = x_j * x_i.transpose();
    kappa = kappa.array().pow(this->degree);
    kappa = this->gamma * kappa;

    return kappa;
}

RBF::RBF() {
    this->name = "rbf";
    this->gamma = 1.0;
}

RBF::RBF(std::string name, double gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd RBF::PyCall(const Eigen::MatrixXd &x_i,
                            const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = (x_j - x_i).array().pow(2);
    kappa = -kappa.rowwise().sum();
    kappa = (this->gamma * kappa).array().exp();

    return kappa;
}