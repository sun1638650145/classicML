//
// losses.cc
// losses
//
// Create by 孙瑞琦 on 2020/1/25.
//
//

#include "losses.h"

Loss::Loss() {
    this->name = "loss";
}

Loss::Loss(std::string name) {
    this->name = std::move(name);
}

double Loss::PyCall(const Eigen::MatrixXd &y_pred,
                    const Eigen::MatrixXd &y_true) {
    throw NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

BinaryCrossentropy::BinaryCrossentropy() {
    this->name = "binary_crossentropy";
}

BinaryCrossentropy::BinaryCrossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致).
double BinaryCrossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                                  const Eigen::MatrixXd &y_true) {
    int y_shape = y_true.rows();

    Eigen::MatrixXd loss_l = ((-y_true.transpose()).adjoint().array() * y_pred.array().log());
    Eigen::MatrixXd loss_r = (1 - y_true.array().transpose().array());
    loss_r = loss_r.adjoint().array() * (1 - y_pred.array()).log();

    return (loss_l.sum() + loss_r.sum()) / y_shape;
}

CategoricalCrossentropy::CategoricalCrossentropy() {
    this->name = "categorical_crossentropy";
}

CategoricalCrossentropy::CategoricalCrossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致).
double CategoricalCrossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                                       const Eigen::MatrixXd &y_true) {
    int y_shape = y_true.rows();

    Eigen::MatrixXd loss = y_true.array() * y_pred.array().log();

    return -loss.sum() / y_shape;
}

Crossentropy::Crossentropy() {
    this->name = "crossentropy";
}

Crossentropy::Crossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致). 将根据标签的实际形状自动使用二分类或者多分类损失函数.
double Crossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                            const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() == 1) {
        BinaryCrossentropy loss = BinaryCrossentropy();
        return loss.PyCall(y_pred, y_true);
    } else {
        CategoricalCrossentropy loss = CategoricalCrossentropy();
        return loss.PyCall(y_pred, y_true);
    }
}