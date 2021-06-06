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

Eigen::MatrixXd activations::Activation::PyCall(const Eigen::MatrixXd &z) {
    throw exceptions::NotImplementedError();
}

Eigen::MatrixXd activations::Activation::Diff(const Eigen::MatrixXd &output,
                                              const Eigen::MatrixXd &a,
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

// 经过激活后的张量, 输入为张量.
Eigen::MatrixXd activations::Relu::PyCall(const Eigen::MatrixXd &z) {
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
Eigen::MatrixXd activations::Relu::Diff(const Eigen::MatrixXd &output,
                                        const Eigen::MatrixXd &a,
                                        const pybind11::args &args,
                                        const pybind11::kwargs &kwargs) {
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

activations::Sigmoid::Sigmoid() {
    this->name = "sigmoid";
}

activations::Sigmoid::Sigmoid(std::string name) {
    this->name = std::move(name);
}

// 经过激活后的张量, 输入为张量.
Eigen::MatrixXd activations::Sigmoid::PyCall(const Eigen::MatrixXd &z) {
    Eigen::MatrixXd result(z.rows(), z.cols());
    result = 1 / (1 + (-z.array()).exp());

    return result;
}

// 计算函数的微分, 输入为输出的张量, 输入的张量和真实的标签.
Eigen::MatrixXd activations::Sigmoid::Diff(const Eigen::MatrixXd &output,
                                           const Eigen::MatrixXd &a,
                                           const pybind11::args &args,
                                           const pybind11::kwargs &kwargs) {
    Eigen::MatrixXd y_true = pybind11::cast<Eigen::MatrixXd>(args[0]);
    Eigen::MatrixXd error = y_true - output;
    Eigen::MatrixXd da = a.array() * (1 - a.array()) * error.array();

    return da;
}

activations::Softmax::Softmax() {
    this->name = "softmax";
}

activations::Softmax::Softmax(std::string name) {
    this->name = std::move(name);
}

// 经过激活后的张量, 输入为张量.
Eigen::MatrixXd activations::Softmax::PyCall(const Eigen::MatrixXd &z) {
    Eigen::MatrixXd temp_z = z;
    Eigen::MatrixXd result = z;

    // 为了避免溢出减去最大值
    temp_z = temp_z.array() - z.maxCoeff();
    temp_z = temp_z.array().exp();

    for (int row = 0; row < z.rows(); row ++) {
        for (int col = 0; col < z.cols(); col ++) {
            result(row, col) = temp_z(row, col) / temp_z.row(row).sum();
        }
    }

    return result;
}

// Softmax函数的微分, 输入为输出的张量, 输入的张量和真实的标签.
Eigen::MatrixXd activations::Softmax::Diff(const Eigen::MatrixXd &output,
                                           const Eigen::MatrixXd &a,
                                           const pybind11::args &args,
                                           const pybind11::kwargs &kwargs) {
    Eigen::MatrixXd da = a - output;

    return da;
}