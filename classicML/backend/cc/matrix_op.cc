//
//  matrix_op.cc
//  matrix_op
//
//  Created by 孙瑞琦 on 2020/12/8.
//

#include "matrix_op.h"

// 返回广播减法的矩阵, 输入a矩阵和b矩阵.
Eigen::MatrixXd matrix_op::BroadcastSub(const Eigen::MatrixXd &a,
                                        const Eigen::MatrixXd &b) {
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
            Eigen::MatrixXd result_matrix(a.rows(), a.cols());
            for (int row = 0; row < a.rows(); row ++) {
                result_matrix.row(row) = a.row(row) - b;
            }

            return result_matrix;
        } else {
            Eigen::MatrixXd result_matrix(b.rows(), b.cols());
            for (int row = 0; row < b.rows(); row ++) {
                result_matrix.row(row) = b.row(row) - a;
            }

            return -result_matrix;
        }
    }
}

// 生成标准随机正态分布矩阵, 输入为矩阵的行数, 列数和随机种子.
Eigen::MatrixXd matrix_op::GenerateRandomStandardNormalDistributionMatrix(const int &rows,
                                                                          const int &columns,
                                                                          const std::optional<unsigned int> &seed) {
    static std::normal_distribution<double> _distribution(0,1);
    static std::default_random_engine _engine;
    if (!seed.has_value()) {
        _engine.seed(time(nullptr));
    } else {
        _engine.seed(seed.value());
    }

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(rows, columns).unaryExpr(
            [] (double element) {
                return _distribution(_engine);
            }
    );

    return matrix;
}

// 生成随机均匀分布矩阵, 输入为矩阵的行数, 列数和随机种子.
Eigen::MatrixXd matrix_op::GenerateRandomUniformDistributionMatrix(const int &rows,
                                                                   const int &columns,
                                                                   const std::optional<unsigned int> &seed) {
    static std::uniform_real_distribution<double> _distribution(0,1);
    static std::default_random_engine _engine;
    if (!seed.has_value()) {
        _engine.seed(time(nullptr));
    } else {
        _engine.seed(seed.value());
    }

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(rows, columns).unaryExpr(
            [] (double element) {
                return _distribution(_engine);
            }
    );

    return matrix;
}

// 获取非零元素组成的子矩阵, 输入父矩阵和非零标签.
Eigen::MatrixXd matrix_op::GetNonZeroSubMatrix(const Eigen::MatrixXd &matrix,
                                               const Eigen::VectorXd &non_zero_mark) {
    if (matrix.rows() != non_zero_mark.rows()) {
        throw pybind11::value_error("行数不同, 无法操作");
    }

    // 初始化子矩阵.
    int row = (int)(non_zero_mark.array() == 1).count();  // 统计非零元素个数.
    Eigen::MatrixXd sub_matrix(row, matrix.cols());

    row = 0;
    for (int i = 0; i < non_zero_mark.rows(); i ++) {
        if (non_zero_mark[i] == 1) {
            sub_matrix.row(row) = matrix.row(i);
            row ++;
        }
    }

    return sub_matrix;
}

// 返回非零元素下标组成的数组, 输入为数组.
std::vector<int> matrix_op::NonZero(const Eigen::RowVectorXd &array) {
    std::vector<int> buffer;

    for (int i = 0; i < array.size(); i ++) {
        if (array[i] != 0.0) {
            buffer.push_back(i);
        }
    }

    return buffer;
}

// 返回一个有相同数据的值的新形状的矩阵, 输入为要改变形状的矩阵, 改变后的行数和列数.
Eigen::MatrixXd matrix_op::Reshape(Eigen::MatrixXd matrix, const int &row, const int &column) {
    int new_row = row;
    int new_column = column;

    // 可以将一个维度指定为-1, 函数将自动推理.
    if (row == -1 && column == -1) {
        throw pybind11::value_error("只能指定维度为一个未知维度");
    } else {
        if (row == -1) {
            new_row = (int)matrix.size() / column;
        } else if (column == -1) {
            new_column = (int)matrix.size() / row;
        }
    }

    // 在官方API中Eigen没有提供reshape方法, 这里是参照官方文档实现的一种代替方式.
    Eigen::Map<Eigen::MatrixXd> map(matrix.data(), new_row, new_column);
    Eigen::MatrixXd reshaped_matrix = map;

    return reshaped_matrix;
}