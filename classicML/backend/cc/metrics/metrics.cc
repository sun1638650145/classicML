//
// metrics.cc
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#include "metrics.h"

namespace metrics {
Metric::Metric() {
    this->name = "metric";
}

Metric::Metric(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
template<typename Dtype, typename Matrix>
Dtype Metric::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    throw exceptions::NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

Accuracy::Accuracy() {
    this->name = "accuracy";
}

Accuracy::Accuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype Accuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    if (y_pred.cols() == 1) {
        BinaryAccuracy metric = BinaryAccuracy();
        return metric.PyCall<Dtype, Matrix>(y_pred, y_true);
    } else {
        CategoricalAccuracy metric = CategoricalAccuracy();
        return metric.PyCall<Dtype, Matrix>(y_pred, y_true);
    }
}

BinaryAccuracy::BinaryAccuracy() {
    this->name = "binary_accuracy";
}

BinaryAccuracy::BinaryAccuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype BinaryAccuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    if (y_pred.cols() != y_true.cols() || y_pred.rows() != y_true.rows()) {
        throw pybind11::value_error("形状不一致");
    }

    auto accuracy = (Dtype)y_pred.size();

    for (int32 i = 0; i < y_pred.size(); i++) {
        if ((y_pred(i, 0) < 0.5 && y_true(i, 0) == 1) ||
            (y_pred(i, 0) >= 0.5 && y_true(i, 0) == 0)) {
            accuracy -= 1;
        }
    }

    return accuracy / (Dtype)y_pred.size();
}

CategoricalAccuracy::CategoricalAccuracy() {
    this->name = "categorical_accuracy";
}

CategoricalAccuracy::CategoricalAccuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype CategoricalAccuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    if (y_pred.cols() != y_true.cols() || y_pred.rows() != y_true.rows()) {
        throw pybind11::value_error("形状不一致");
    }

    auto accuracy = (Dtype)y_pred.rows();

    for (int32 row = 0; row < y_pred.rows(); row++) {
        int32 y_pred_row, y_pred_col;
        int32 y_true_row, y_true_col;

        y_pred.row(row).maxCoeff(&y_pred_row, &y_pred_col);
        y_true.row(row).maxCoeff(&y_true_row, &y_true_col);

        if (y_true_col != y_pred_col) {
            accuracy -= 1;
        }
    }

    return accuracy / (Dtype)y_pred.rows();
}

// 显式实例化.
template float32 Metric::PyCall(const matrix32 &y_pred, const matrix32 &y_true);
template float64 Metric::PyCall(const matrix64 &y_pred, const matrix64 &y_true);

template float32 Accuracy::PyCall(const matrix32 &y_pred, const matrix32 &y_true);
template float64 Accuracy::PyCall(const matrix64 &y_pred, const matrix64 &y_true);

template float32 BinaryAccuracy::PyCall(const matrix32 &y_pred, const matrix32 &y_true);
template float64 BinaryAccuracy::PyCall(const matrix64 &y_pred, const matrix64 &y_true);

template float32 CategoricalAccuracy::PyCall(const matrix32 &y_pred, const matrix32 &y_true);
template float64 CategoricalAccuracy::PyCall(const matrix64 &y_pred, const matrix64 &y_true);
}  // namespace metrics