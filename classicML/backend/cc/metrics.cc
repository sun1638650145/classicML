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

double metrics::Metric::PyCall(const Eigen::MatrixXd &y_pred,
                               const Eigen::MatrixXd &y_true) {
    throw exceptions::NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

metrics::Accuracy::Accuracy() {
    this->name = "accuracy";
}

metrics::Accuracy::Accuracy(std::string name) {
    this->name = std::move(name);
}

// 返回准确率, 输入为两个张量(两个向量必须形状一致).
double metrics::Accuracy::PyCall(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() == 1) {
        BinaryAccuracy metric = BinaryAccuracy();
        return metric.PyCall(y_pred, y_true);
    } else {
        CategoricalAccuracy metric = CategoricalAccuracy();
        return metric.PyCall(y_pred, y_true);
    }
}

metrics::BinaryAccuracy::BinaryAccuracy() {
    this->name = "binary_accuracy";
}

metrics::BinaryAccuracy::BinaryAccuracy(std::string name) {
    this->name = std::move(name);
}

// 返回准确率, 输入为两个张量(两个向量必须形状一致).
double metrics::BinaryAccuracy::PyCall(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() != y_true.cols() || y_pred.rows() != y_true.rows()) {
        throw pybind11::value_error("形状不一致");
    }

    auto accuracy = (double)y_pred.size();

    for (int i = 0; i < y_pred.size(); i++) {
        if ((y_pred(i, 0) < 0.5 && y_true(i, 0) == 1) ||
            (y_pred(i, 0) >= 0.5 && y_true(i, 0) == 0)) {
            accuracy -= 1;
        }
    }

    return accuracy / (double)y_pred.size();
}

metrics::CategoricalAccuracy::CategoricalAccuracy() {
    this->name = "categorical_accuracy";
}

metrics::CategoricalAccuracy::CategoricalAccuracy(std::string name) {
    this->name = std::move(name);
}

// 返回准确率, 输入为两个张量(两个向量必须形状一致).
double metrics::CategoricalAccuracy::PyCall(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() != y_true.cols() || y_pred.rows() != y_true.rows()) {
        throw pybind11::value_error("形状不一致");
    }

    auto accuracy = (double)y_pred.rows();

    for (int row = 0; row < y_pred.rows(); row++) {
        int y_pred_row, y_pred_col;
        int y_true_row, y_true_col;

        y_pred.row(row).maxCoeff(&y_pred_row, &y_pred_col);
        y_true.row(row).maxCoeff(&y_true_row, &y_true_col);

        if (y_true_col != y_pred_col) {
            accuracy -= 1;
        }
    }

    return accuracy / (double)y_pred.rows();
}
