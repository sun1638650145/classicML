//
// metrics.cc
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#include "metrics.h"

metrics::Metric::Metric() {
    this->name = "metric";
}

metrics::Metric::Metric(std::string name) {
    this->name = std::move(name);
}


// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
template<typename Dtype, typename Matrix>
Dtype metrics::Metric::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    throw exceptions::NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

metrics::Accuracy::Accuracy() {
    this->name = "accuracy";
}

metrics::Accuracy::Accuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype metrics::Accuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
    if (y_pred.cols() == 1) {
        BinaryAccuracy metric = BinaryAccuracy();
        return metric.PyCall<Dtype, Matrix>(y_pred, y_true);
    } else {
        CategoricalAccuracy metric = CategoricalAccuracy();
        return metric.PyCall<Dtype, Matrix>(y_pred, y_true);
    }
}

metrics::BinaryAccuracy::BinaryAccuracy() {
    this->name = "binary_accuracy";
}

metrics::BinaryAccuracy::BinaryAccuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype metrics::BinaryAccuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
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

metrics::CategoricalAccuracy::CategoricalAccuracy() {
    this->name = "categorical_accuracy";
}

metrics::CategoricalAccuracy::CategoricalAccuracy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回准确率, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype metrics::CategoricalAccuracy::PyCall(const Matrix &y_pred, const Matrix &y_true) {
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