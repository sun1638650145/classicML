//
// kernels.cc
// kernels
//
// Create by 孙瑞琦 on 2020/2/10.
//
//

#include "kernels.h"

kernels::Kernel::Kernel() {
    this->name = "kernel";
}

kernels::Kernel::Kernel(std::string name) {
    this->name = std::move(name);
}

Eigen::MatrixXd kernels::Kernel::PyCall(const Eigen::MatrixXd &x_i,
                                        const Eigen::MatrixXd &x_j) {
    throw NotImplementedError();  // 与Py后端实现相同, 主动抛出异常.
}

kernels::Linear::Linear() {
    this->name = "linear";
}

kernels::Linear::Linear(std::string name) {
    this->name = std::move(name);
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd kernels::Linear::PyCall(const Eigen::MatrixXd &x_i,
                                        const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = x_j * x_i.transpose();

    return kappa;
}

kernels::Polynomial::Polynomial() {
    this->name = "poly";
    this->gamma = 1.0;
    this->degree = 3;
}

kernels::Polynomial::Polynomial(std::string name, double gamma, int degree) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->degree = degree;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd kernels::Polynomial::PyCall(const Eigen::MatrixXd &x_i,
                                            const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = x_j * x_i.transpose();
    kappa = kappa.array().pow(this->degree);
    kappa = this->gamma * kappa;

    return kappa;
}

kernels::RBF::RBF() {
    this->name = "rbf";
    this->gamma = 1.0;
}

kernels::RBF::RBF(std::string name, double gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd kernels::RBF::PyCall(const Eigen::MatrixXd &x_i,
                                     const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = (x_j - x_i).array().pow(2);
    kappa = -kappa.rowwise().sum();
    kappa = (this->gamma * kappa).array().exp();

    return kappa;
}

kernels::Gaussian::Gaussian() {
    this->name = "gaussian";
    this->gamma = 1.0;
}

kernels::Gaussian::Gaussian(std::string name, double gamma) {
    this->name = std::move(name);
    this->gamma = gamma;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd kernels::Gaussian::PyCall(const Eigen::MatrixXd &x_i,
                                          const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = (x_j - x_i).array().pow(2);
    kappa = -kappa.rowwise().sum();
    kappa = (this->gamma * kappa).array().exp();

    return kappa;
}

kernels::Sigmoid::Sigmoid() {
    this->name = "sigmoid";
    this->gamma = 1.0;
    this->beta = 1.0;
    this->theta = -1.0;
}

kernels::Sigmoid::Sigmoid(std::string name, double gamma, double beta, double theta) {
    this->name = std::move(name);
    this->gamma = gamma;
    this->beta = beta;
    this->theta = theta;
}

// 返回核函数映射后的特征向量, 输入为两组特征张量(两个张量的形状必须一致).
Eigen::MatrixXd kernels::Sigmoid::PyCall(const Eigen::MatrixXd &x_i,
                                         const Eigen::MatrixXd &x_j) {
    Eigen::MatrixXd kappa = this->beta * (x_j * x_i.transpose()).array() + this->theta;
    kappa = kappa.array().tanh();
    kappa = this->gamma * kappa;

    return kappa;
}