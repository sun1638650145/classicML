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

initializers::Initializer::Initializer(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix initializers::Initializer::PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}

initializers::RandomNormal::RandomNormal() {
    this->name = "random_normal";
}

initializers::RandomNormal::RandomNormal(std::string name) {
    this->name = std::move(name);
}

initializers::RandomNormal::RandomNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是int32/int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Dtype` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Dtype>
std::map<std::string, Matrix> initializers::RandomNormal::PyCall(const RowVector &attributes_or_structure) {
    std::map<std::string, Matrix> parameters;
    Matrix w, b;

    auto num_of_layers = (int32)attributes_or_structure.size();
    for (int32 layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Dtype>
                (attributes_or_structure[layer + 1], attributes_or_structure[layer], this->seed);
        b = Matrix::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(int32/int64).
// References: [Efficient arrays of numeric values](https://docs.python.org/3.8/library/array.html)
std::variant<Eigen::MatrixXf, Eigen::MatrixXd>
initializers::RandomNormal::PyCall(const pybind11::buffer &attributes_or_structure) {
    std::string type_code = attributes_or_structure.request().format;
    if (type_code == "i") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
                (pybind11::cast<int32>(attributes_or_structure) + 1, 1, this->seed);
        return parameters;
    } else if (type_code == "l") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXd, float64>
                (pybind11::cast<int32>(attributes_or_structure) + 1, 1, this->seed);
        return parameters;
    }

    return {};
}

// 初始化参数矩阵(32位), 输入为一个整数(Pure Python int).
Eigen::MatrixXf initializers::RandomNormal::PyCall(const int32 &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
            ((int32)attributes_or_structure + 1, 1, this->seed);

    return parameters;
}

initializers::HeNormal::HeNormal() {
    this->name = "he_normal";
}

initializers::HeNormal::HeNormal(std::string name) {
    this->name = std::move(name);
}

initializers::HeNormal::HeNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是int32/int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Dtype` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Dtype>
std::map<std::string, Matrix> initializers::HeNormal::PyCall(const RowVector &attributes_or_structure) {
    std::map<std::string, Matrix> parameters;
    Matrix w, b;

    auto num_of_layers = (int32)attributes_or_structure.size();
    for (int32 layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Dtype>
                (attributes_or_structure[layer + 1], attributes_or_structure[layer], this->seed);
        w = w * sqrt(2.0 / attributes_or_structure[layer]);

        b = Matrix::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(int32/int64).
std::variant<Eigen::MatrixXf, Eigen::MatrixXd>
initializers::HeNormal::PyCall(const pybind11::buffer &attributes_or_structure) {
    std::string type_code = attributes_or_structure.request().format;
    if (type_code == "i") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
                (pybind11::cast<int32>(attributes_or_structure) + 1, 1, this->seed);
        parameters = parameters * sqrt(2.0 / pybind11::cast<int32>(attributes_or_structure));

        return parameters;
    } else if (type_code == "l") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXd, float64>
                (pybind11::cast<int32>(attributes_or_structure) + 1, 1, this->seed);
        parameters = parameters * sqrt(2.0 / pybind11::cast<int32>(attributes_or_structure));

        return parameters;
    }

    return {};
}

// 初始化参数矩阵(32位), 输入为一个整数(Pure Python int).
Eigen::MatrixXf initializers::HeNormal::PyCall(const int32 &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
            (attributes_or_structure + 1, 1, this->seed);
    parameters = parameters * sqrt(2.0 / attributes_or_structure);

    return parameters;
}

initializers::XavierNormal::XavierNormal() {
    this->name = "xavier_normal";
}

initializers::XavierNormal::XavierNormal(std::string name) {
    this->name = std::move(name);
}

initializers::XavierNormal::XavierNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是int32/int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Dtype` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Dtype>
std::map<std::string, Matrix> initializers::XavierNormal::PyCall(const RowVector &attributes_or_structure) {
    std::map<std::string, Matrix> parameters;
    Matrix w, b;

    auto num_of_layers = (int32)attributes_or_structure.size();
    for (int32 layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Dtype>
                (attributes_or_structure[layer + 1], attributes_or_structure[layer], this->seed);
        w = w * sqrt(2.0 / (attributes_or_structure[layer] + attributes_or_structure[layer + 1]));

        b = Matrix::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(int32/int64).
std::variant<Eigen::MatrixXf, Eigen::MatrixXd>
initializers::XavierNormal::PyCall(const pybind11::buffer &attributes_or_structure) {
    std::string type_code = attributes_or_structure.request().format;
    if (type_code == "i") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
                (pybind11::cast<int32>(attributes_or_structure), 1, this->seed);
        parameters = parameters * sqrt(pybind11::cast<float32>(attributes_or_structure));

        return parameters;
    } else if (type_code == "l") {
        auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXd, float64>
                (pybind11::cast<int32>(attributes_or_structure), 1, this->seed);
        parameters = parameters * sqrt(pybind11::cast<float64>(attributes_or_structure));

        return parameters;
    }

    return {};
}

// 初始化参数矩阵(32位), 输入为一个整数(Pure Python int).
Eigen::MatrixXf initializers::XavierNormal::PyCall(const int32 &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
            (attributes_or_structure + 1, 1, this->seed);
    parameters = parameters * sqrt((float32)attributes_or_structure);

    return parameters;
}

initializers::GlorotNormal::GlorotNormal() {
    this->name = "glorot_normal";
}

initializers::GlorotNormal::GlorotNormal(std::string name) {
    this->name = std::move(name);
}

initializers::GlorotNormal::GlorotNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

initializers::RBFNormal::RBFNormal() {
    this->name = "rbf_normal";
}

initializers::RBFNormal::RBFNormal(std::string name) {
    this->name = std::move(name);
}

initializers::RBFNormal::RBFNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(int32/int64).
std::variant<std::map<std::string, Eigen::MatrixXf>, std::map<std::string, Eigen::MatrixXd>>
initializers::RBFNormal::PyCall(const pybind11::buffer &hidden_units) {
    std::string type_code = hidden_units.request().format;
    if (type_code == "i") {
        std::map<std::string, Eigen::MatrixXf> parameters;

        parameters["w"] = Eigen::MatrixXf::Zero(1, pybind11::cast<int32>(hidden_units));
        parameters["b"] = Eigen::MatrixXf::Zero(1, 1);
        parameters["c"] = matrix_op::GenerateRandomUniformDistributionMatrix<Eigen::MatrixXf, float32>
                (pybind11::cast<int32>(hidden_units), 2, seed);
        parameters["beta"] = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
                (1, pybind11::cast<int32>(hidden_units), seed);

        return parameters;
    } else if (type_code == "l") {
        std::map<std::string, Eigen::MatrixXd> parameters;

        parameters["w"] = Eigen::MatrixXd::Zero(1, pybind11::cast<int32>(hidden_units));
        parameters["b"] = Eigen::MatrixXd::Zero(1, 1);
        parameters["c"] = matrix_op::GenerateRandomUniformDistributionMatrix<Eigen::MatrixXd, float64>
                (pybind11::cast<int32>(hidden_units), 2, seed);
        parameters["beta"] = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXd, float64>
                (1, pybind11::cast<int32>(hidden_units), seed);

        return parameters;
    }

    return {};
}

// 初始化参数矩阵(32位), 输入为一个整数(Pure Python int).
std::map<std::string, Eigen::MatrixXf> initializers::RBFNormal::PyCall(const int32 &hidden_units) {
    std::map<std::string, Eigen::MatrixXf> parameters;

    parameters["w"] = Eigen::MatrixXf::Zero(1, hidden_units);
    parameters["b"] = Eigen::MatrixXf::Zero(1, 1);
    parameters["c"] = matrix_op::GenerateRandomUniformDistributionMatrix<Eigen::MatrixXf, float32>(hidden_units, 2, seed);
    parameters["beta"] = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
                             (1, hidden_units, seed);

    return parameters;
}