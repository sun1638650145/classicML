//
// initializers.cc
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
// Refactor by 孙瑞琦 on 2021/12/25.
//
//

#include "initializers.h"

namespace initializers {
Initializer::Initializer() {
    this->name = "initializer";
}

Initializer::Initializer(std::string name) {
    this->name = std::move(name);
}

Initializer::Initializer(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix Initializer::PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}

RandomNormal::RandomNormal() {
    this->name = "random_normal";
}

RandomNormal::RandomNormal(std::string name) {
    this->name = std::move(name);
}

RandomNormal::RandomNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(pure int/np.int32/np.int64).
template<typename Matrix, typename Int, typename Float>
Matrix RandomNormal::PyCall1(Int &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
            ((int32)attributes_or_structure + 1, 1, this->seed);

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是np.int32/np.int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Float` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Float>
std::map<std::string, Matrix> RandomNormal::PyCall2(const RowVector &attributes_or_structure) {
    std::map<std::string, Matrix> parameters;
    Matrix w, b;

    auto num_of_layers = (int32)attributes_or_structure.size();
    for (int32 layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
                (attributes_or_structure[layer + 1], attributes_or_structure[layer], this->seed);
        b = Matrix::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}

HeNormal::HeNormal() {
    this->name = "he_normal";
}

HeNormal::HeNormal(std::string name) {
    this->name = std::move(name);
}

HeNormal::HeNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(pure int/np.int32/np.int64).
template<typename Matrix, typename Int, typename Float>
Matrix HeNormal::PyCall1(Int &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
                (attributes_or_structure + 1, 1, this->seed);
    parameters = parameters * sqrt(2.0 / attributes_or_structure);

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是np.int32/np.int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Float` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Dtype>
std::map<std::string, Matrix> HeNormal::PyCall2(const RowVector &attributes_or_structure) {
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

XavierNormal::XavierNormal() {
    this->name = "xavier_normal";
}

XavierNormal::XavierNormal(std::string name) {
    this->name = std::move(name);
}

XavierNormal::XavierNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(pure int/np.int32/np.int64).
template<typename Matrix, typename Int, typename Float>
Matrix XavierNormal::PyCall1(Int &attributes_or_structure) {
    auto parameters = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
        (attributes_or_structure + 1, 1, this->seed);
    parameters = parameters * sqrt((Float)attributes_or_structure);

    return parameters;
}

// 初始化参数矩阵(32/64位), 输入为一个列表(元素是np.int32/np.int64).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `RowVector` 兼容32位和64位整数行向量, `Float` 兼容32位和64位浮点数;
// 不支持不同位数模板兼容.
template<typename Matrix, typename RowVector, typename Float>
std::map<std::string, Matrix> XavierNormal::PyCall2(const RowVector &attributes_or_structure) {
    std::map<std::string, Matrix> parameters;
    Matrix w, b;

    auto num_of_layers = (int32)attributes_or_structure.size();
    for (int32 layer = 0; layer < num_of_layers - 1; layer ++) {
        w = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
            (attributes_or_structure[layer + 1], attributes_or_structure[layer], this->seed);
        w = w * sqrt(2.0 / (attributes_or_structure[layer] + attributes_or_structure[layer + 1]));

        b = Matrix::Zero(1, attributes_or_structure[layer + 1]);

        parameters["w" + std::to_string(layer + 1)] = w;
        parameters["b" + std::to_string(layer + 1)] = b;
    }

    return parameters;
}

GlorotNormal::GlorotNormal() {
    this->name = "glorot_normal";
}

GlorotNormal::GlorotNormal(std::string name) {
    this->name = std::move(name);
}

GlorotNormal::GlorotNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

RBFNormal::RBFNormal() {
    this->name = "rbf_normal";
}

RBFNormal::RBFNormal(std::string name) {
    this->name = std::move(name);
}

RBFNormal::RBFNormal(std::string name, std::optional<uint32> seed) {
    this->name = std::move(name);
    this->seed = seed;
}

// 初始化参数矩阵(32/64位), 输入为一个整数(pure int/np.int32/np.int64).
template<typename Matrix, typename Int, typename Float>
std::map<std::string, Matrix> RBFNormal::PyCall(Int &hidden_units) {
    std::map<std::string, Matrix> parameters;

    parameters["w"] = Matrix::Zero(1, hidden_units);
    parameters["b"] = Matrix::Zero(1, 1);
    parameters["c"] = matrix_op::GenerateRandomUniformDistributionMatrix<Matrix, Float>(hidden_units, 2, seed);
    parameters["beta"] = matrix_op::GenerateRandomStandardNormalDistributionMatrix<Matrix, Float>
        (1, hidden_units, seed);

    return parameters;
}

// 显式实例化.
template matrix32 Initializer::PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs);
template matrix64 Initializer::PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs);

template matrix32 RandomNormal::PyCall1<matrix32, np_int32, float32>(np_int32 &attributes_or_structure);
template matrix64 RandomNormal::PyCall1<matrix64, np_int64, float64>(np_int64 &attributes_or_structure);
template matrix32 RandomNormal::PyCall1<matrix32, int32, float32>(int32 &attributes_or_structure);
template std::map<std::string, matrix32> RandomNormal::PyCall2<matrix32, row_vector32i, float32>
        (const row_vector32i &attributes_or_structure);
template std::map<std::string, matrix64> RandomNormal::PyCall2<matrix64, row_vector64i, float64>
        (const row_vector64i &attributes_or_structure);

template matrix32 HeNormal::PyCall1<matrix32, np_int32, float32>(np_int32 &attributes_or_structure);
template matrix64 HeNormal::PyCall1<matrix64, np_int64, float64>(np_int64 &attributes_or_structure);
template matrix32 HeNormal::PyCall1<matrix32, int32, float32>(int32 &attributes_or_structure);
template std::map<std::string, matrix32> HeNormal::PyCall2<matrix32, row_vector32i, float32>
        (const row_vector32i &attributes_or_structure);
template std::map<std::string, matrix64> HeNormal::PyCall2<matrix64, row_vector64i, float64>
        (const row_vector64i &attributes_or_structure);

template matrix32 XavierNormal::PyCall1<matrix32, np_int32, float32>(np_int32 &attributes_or_structure);
template matrix64 XavierNormal::PyCall1<matrix64, np_int64, float64>(np_int64 &attributes_or_structure);
template matrix32 XavierNormal::PyCall1<matrix32, int32, float32>(int32 &attributes_or_structure);
template std::map<std::string, matrix32> XavierNormal::PyCall2<matrix32, row_vector32i, float32>
        (const row_vector32i &attributes_or_structure);
template std::map<std::string, matrix64> XavierNormal::PyCall2<matrix64, row_vector64i, float64>
        (const row_vector64i &attributes_or_structure);

template std::map<std::string, matrix32> RBFNormal::PyCall<matrix32, np_int32, float32>(np_int32 &hidden_units);
template std::map<std::string, matrix64> RBFNormal::PyCall<matrix64, np_int64, float64>(np_int64 &hidden_units);
template std::map<std::string, matrix32> RBFNormal::PyCall<matrix32, int32, float32>(int32 &hidden_units);
} // namespace initializers