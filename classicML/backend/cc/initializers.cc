//
// initializers.cc
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
//
//

#include "initializers.h"

initializers::Initializer::Initializer() {
    this->name = "initializer";
}

initializers::Initializer::Initializer(std::string name) {
    this->name = std::move(name);
}

initializers::Initializer::Initializer(std::string name, std::optional<unsigned int> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

Eigen::MatrixXd initializers::Initializer::PyCall(const pybind11::args &args,
                                                  const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}

initializers::RandomNormal::RandomNormal() {
    this->name = "random_normal";
}

initializers::RandomNormal::RandomNormal(std::string name) {
    this->name = std::move(name);
}

initializers::RandomNormal::RandomNormal(std::string name, std::optional<unsigned int> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化的参数矩阵, 输入为一个整数.
Eigen::MatrixXd initializers::RandomNormal::PyCall(const int &attributes_or_structure) {
    Eigen::MatrixXd parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix(attributes_or_structure + 1,
                                                                                           1,
                                                                                           this->seed);

    return parameters;
}

// 初始化的参数矩阵, 输入为一个列表.
std::map<std::string, Eigen::MatrixXd> initializers::RandomNormal::PyCall(const Eigen::RowVectorXi &attributes_or_structure) {
    std::map<std::string, Eigen::MatrixXd> parameters;
    Eigen::MatrixXd w, b;

    int num_of_layers = (int)attributes_or_structure.size();
    for (int layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix(attributes_or_structure[layer + 1],
                                                                      attributes_or_structure[layer],
                                                                      this->seed);
        b = Eigen::MatrixXd::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}