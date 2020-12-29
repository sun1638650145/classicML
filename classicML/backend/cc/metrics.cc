//
// metrics.cc
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#include "metrics.h"

Metric::Metric() {
    this->name = "metric";
}

Metric::Metric(std::string name) {
    this->name = std::move(name);
}

double Metric::PyCall(const Eigen::MatrixXd &y_pred,
                      const Eigen::MatrixXd &y_true) {
    throw NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

BinaryAccuracy::BinaryAccuracy() {
    this->name = "binary_accuracy";
}

BinaryAccuracy::BinaryAccuracy(std::string name) {
    this->name = std::move(name);
}

// 返回准确率, 输入为两个张量(两个向量必须形状一致).
double BinaryAccuracy::PyCall(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() != y_true.cols() || y_pred.rows() != y_true.rows()) {
        throw pybind11::value_error("形状不一致");
    }

    double accuracy = y_pred.size();

    for (int i = 0; i < y_pred.size(); i ++) {
        if ((y_pred(i, 0) < 0.5 && y_true(i, 0) == 1) ||
            (y_pred(i, 0) >= 0.5 && y_true(i, 0) == 0)) {
            accuracy -= 1;
        }
    }

    return accuracy / y_pred.size();
}