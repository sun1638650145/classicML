//
// activations.cc
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#include "activations.h"

Activation::Activation() {
    this->name = "activation";
}

Activation::Activation(std::string name) {
    this->name = std::move(name);
}

Eigen::MatrixXd Activation::PyCall(const Eigen::MatrixXd &z) {
    throw NotImplementedError();
}

Eigen::MatrixXd Activation::Diff(const Eigen::MatrixXd &output, const Eigen::MatrixXd &a) {
    throw NotImplementedError();
}

Relu::Relu() {
    this->name = "relu";
}

Relu::Relu(std::string name) {
    this->name = std::move(name);
}

// 经过激活后的张量, 输入为张量.
Eigen::MatrixXd Relu::PyCall(const Eigen::MatrixXd &z) {
    Eigen::MatrixXd result(z.rows(), z.cols());

    for (int row = 0; row < z.rows(); row ++) {
        for (int col = 0; col < z.cols(); col ++) {
            if (0 >= z(row, col)) {
                result(row, col) = 0;
            } else {
                result(row, col) = z(row, col);
            }
        }
    }

    return result;
}

// 计算函数的微分, 输入为前向传播输出的张量和输入的张量.
Eigen::MatrixXd Relu::Diff(const Eigen::MatrixXd &output, const Eigen::MatrixXd &a) {
    Eigen::MatrixXd da = output;

    for (int row = 0; row < a.rows(); row ++) {
        for (int col = 0; col < a.cols(); col ++) {
            if (a(row, col) <= 0) {
                da(row, col) = 0;
            }
        }
    }

    return da;
}