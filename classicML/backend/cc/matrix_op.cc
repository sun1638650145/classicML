//
//  matrix_op.cc
//  matrix_op
//
//  Created by 孙瑞琦 on 2020/12/8.
//

#include "matrix_op.h"

// 返回全部元素是否为离散整数的布尔值, 输入为`numpy.ndarray`.
bool matrix_op::AnyDiscreteInteger(const pybind11::array &array) {
    auto array_ = pybind11::cast<Eigen::MatrixXf>(array);

    // 任一元素取整不等于它本身的就是连续值.
    for (int32 row = 0; row < array_.rows(); row ++) {
        for (int32 col = 0; col < array_.cols(); col ++) {
            if (array_(row, col) != (int32)array_(row, col)) {
                return false;
            }
        }
    }

    return true;
}

// 返回广播减法的矩阵, 输入a矩阵和b矩阵.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template <typename Matrix>
Matrix matrix_op::BroadcastSub(const Matrix &a, const Matrix &b) {
    if (a.rows() == b.rows() && a.cols() == b.cols()) {
        // 执行普通减法.
        return a - b;
    } else {
        // 执行广播减法.
        if (a.rows() != 1 && b.rows() != 1) {
            throw pybind11::value_error("张量无法广播");
        }
        if (a.cols() != b.cols()) {
            throw pybind11::value_error("列数不同, 无法操作");
        }

        if (a.rows() > b.rows()) {
            Matrix result_matrix(a.rows(), a.cols());
            for (int row = 0; row < a.rows(); row ++) {
                result_matrix.row(row) = a.row(row) - b;
            }

            return result_matrix;
        } else {
            Matrix result_matrix(b.rows(), b.cols());
            for (int row = 0; row < b.rows(); row ++) {
                result_matrix.row(row) = b.row(row) - a;
            }

            return -result_matrix;
        }
    }
}

// 生成标准随机正态分布矩阵, 输入为矩阵的行数, 列数和随机种子.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵; `Dtype` 兼容32位和64位浮点数.
template<typename Matrix, typename Dtype>
Matrix matrix_op::GenerateRandomStandardNormalDistributionMatrix(const int32 &rows,
                                                                 const int32 &columns,
                                                                 const std::optional<uint32> &seed) {
    static std::normal_distribution<Dtype> _distribution(0,1);
    static std::default_random_engine _engine;
    if (!seed.has_value()) {
        _engine.seed(time(nullptr));
    } else {
        _engine.seed(seed.value());
    }

    Matrix matrix = Matrix::Zero(rows, columns).unaryExpr(
        [](Dtype element) {
            return _distribution(_engine);
        }
    );

    return matrix;
}

// 生成随机均匀分布矩阵, 输入为矩阵的行数, 列数和随机种子.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵; `Dtype` 兼容32位和64位浮点数.
template<typename Matrix, typename Dtype>
Matrix matrix_op::GenerateRandomUniformDistributionMatrix(const int32 &rows,
                                                          const int32 &columns,
                                                          const std::optional<uint32> &seed) {
    static std::uniform_real_distribution<Dtype> _distribution(0,1);
    static std::default_random_engine _engine;
    if (!seed.has_value()) {
        _engine.seed(time(nullptr));
    } else {
        _engine.seed(seed.value());
    }

    Matrix matrix = Matrix::Zero(rows, columns).unaryExpr(
        [](Dtype element) {
            return _distribution(_engine);
        }
    );

    return matrix;
}

// 获取非零元素组成的子矩阵, 输入父矩阵和非零标签.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `Vector` 兼容32位和64位浮点型向量, 不支持不同位数模板兼容.
template<typename Matrix, typename Vector>
Matrix matrix_op::GetNonZeroSubMatrix(const Matrix &matrix, const Vector &non_zero_mark) {
    if (matrix.rows() != non_zero_mark.rows()) {
        throw pybind11::value_error("行数不同, 无法操作");
    }

    // 初始化子矩阵.
    auto row = (int32)(non_zero_mark.array() == 1).count();  // 统计非零元素个数.
    Matrix sub_matrix(row, matrix.cols());

    row = 0;
    for (int32 i = 0; i < non_zero_mark.rows(); i ++) {
        if (non_zero_mark[i] == 1) {
            sub_matrix.row(row) = matrix.row(i);
            row ++;
        }
    }

    return sub_matrix;
}

// 返回非零元素下标组成的数组, 输入为数组.
// `RowVector` 兼容32位和64位浮点型和整型行向量.
template<typename RowVector>
std::vector<uint32> matrix_op::NonZero(const RowVector &array) {
    std::vector<uint32> buffer;

    for (int32 i = 0; i < array.size(); i ++) {
        if (array[i] != 0.0) {
            buffer.push_back(i);
        }
    }

    return buffer;
}

// 返回一个有相同数据的值的新形状的矩阵, 输入为要改变形状的矩阵, 改变后的行数和列数.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵; `Dtype` 兼容32位和64位整型.
// tips: `matrix_op::Reshape` 的内部实现依赖于Eigen::Map, 所以不要惊讶为什么此处的第一个参数不是常引用.
template <typename Matrix, typename Dtype>
Matrix matrix_op::Reshape(Matrix matrix, const Dtype &row, const Dtype &column) {
    Dtype new_row = row;
    Dtype new_column = column;

    // 可以将一个维度指定为-1, 函数将自动推理.
    if (row == -1 && column == -1) {
        throw pybind11::value_error("只能指定维度为一个未知维度");
    } else {
        if (row == -1) {
            new_row = (Dtype)matrix.size() / column;
        } else if (column == -1) {
            new_column = (Dtype)matrix.size() / row;
        }
    }

    // 在官方API中Eigen没有提供reshape方法, 这里是参照官方文档实现的一种代替方式.
    Eigen::Map<Matrix> map(matrix.data(), new_row, new_column);
    Matrix reshaped_matrix = map;

    return reshaped_matrix;
}

// 返回变体(唯一值组成的集合), 输入为`numpy.ndarray`.
std::variant<std::set<float32>, std::set<uint8>> matrix_op::Unique(const pybind11::array &array) {
    if (array.dtype().kind() == 'f' or array.dtype().kind() == 'i') {
        std::set<float32> buffer;
        auto array_ = pybind11::cast<Eigen::MatrixXf>(array);

        for (int row = 0; row < array_.rows(); row ++) {
            for (int col = 0; col < array_.cols(); col ++) {
                buffer.insert(array_(row, col));
            }
        }

        return buffer;
    } else if (array.dtype().kind() == 'U') {
        std::set<uint8> buffer;
        if (array.ndim() == 1) {
            for (int32 i = 0; i != array.size(); i ++) {
                buffer.insert(*(uint8*)array.data(i));
            }
        } else {
            for (int32 i = 0; i != array.shape(0); i ++) {
                for (int32 j = 0; j != array.shape(1); j ++) {
                    buffer.insert(*(uint8*)array.data(i, j));
                }
            }
        }

        return buffer;
    }

    return {};
}

// 显式实例化.
template Eigen::MatrixXf matrix_op::BroadcastSub(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b);
template Eigen::MatrixXd matrix_op::BroadcastSub(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

template Eigen::MatrixXf matrix_op::Reshape(Eigen::MatrixXf matrix, const int32 &row, const int32 &column);
// reshape的row, column 提供的是单精度的情况.
template Eigen::MatrixXd matrix_op::Reshape(Eigen::MatrixXd matrix, const int32 &row, const int32 &column);
template Eigen::MatrixXd matrix_op::Reshape(Eigen::MatrixXd matrix, const int64 &row, const int64 &column);

template Eigen::MatrixXf matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXf, float32>
        (const int32 &rows, const int32 &columns, const std::optional<uint32> &seed);
template Eigen::MatrixXd matrix_op::GenerateRandomStandardNormalDistributionMatrix<Eigen::MatrixXd, float64>
        (const int32 &rows, const int32 &columns, const std::optional<uint32> &seed);

template Eigen::MatrixXf matrix_op::GenerateRandomUniformDistributionMatrix<Eigen::MatrixXf, float32>
        (const int32 &rows, const int32 &columns, const std::optional<uint32> &seed);
template Eigen::MatrixXd matrix_op::GenerateRandomUniformDistributionMatrix<Eigen::MatrixXd, float64>
        (const int32 &rows, const int32 &columns, const std::optional<uint32> &seed);

template Eigen::MatrixXf matrix_op::GetNonZeroSubMatrix(const Eigen::MatrixXf &matrix,
                                                        const Eigen::VectorXf &non_zero_mark);
template Eigen::MatrixXd matrix_op::GetNonZeroSubMatrix(const Eigen::MatrixXd &matrix,
                                                        const Eigen::VectorXd &non_zero_mark);

template std::vector<uint32> matrix_op::NonZero(const Eigen::RowVectorXf &array);
template std::vector<uint32> matrix_op::NonZero(const Eigen::RowVectorXd &array);
template std::vector<uint32> matrix_op::NonZero(const Eigen::RowVectorXi &array);
template std::vector<uint32> matrix_op::NonZero(const Eigen::Matrix<int64, 1, -1> &array);