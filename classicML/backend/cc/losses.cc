//
// losses.cc
// losses
//
// Create by 孙瑞琦 on 2020/1/25.
//
//

#include "losses.h"

loss::Loss::Loss() {
    this->name = "loss";
}

loss::Loss::Loss(std::string name) {
    this->name = std::move(name);
}

double loss::Loss::PyCall(const Eigen::MatrixXd &y_pred,
                          const Eigen::MatrixXd &y_true) {
    throw NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

loss::BinaryCrossentropy::BinaryCrossentropy() {
    this->name = "binary_crossentropy";
}

loss::BinaryCrossentropy::BinaryCrossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致).
double loss::BinaryCrossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                                        const Eigen::MatrixXd &y_true) {
    int y_shape = y_true.rows();

    Eigen::MatrixXd loss_l = ((y_true.transpose()).adjoint().array() * y_pred.array().log());
    Eigen::MatrixXd loss_r = (1 - y_true.array().transpose().array());
    loss_r = loss_r.adjoint().array() * (1 - y_pred.array()).log();

    return -(loss_l.sum() + loss_r.sum()) / y_shape;
}

loss::CategoricalCrossentropy::CategoricalCrossentropy() {
    this->name = "categorical_crossentropy";
}

loss::CategoricalCrossentropy::CategoricalCrossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致).
double loss::CategoricalCrossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                                             const Eigen::MatrixXd &y_true) {
    int y_shape = y_true.rows();
    Eigen::MatrixXd temp = y_pred.array().log();
    Eigen::MatrixXd loss = y_true.array() * temp.array();

    return -loss.sum() / y_shape;
}

loss::Crossentropy::Crossentropy() {
    this->name = "crossentropy";
}

loss::Crossentropy::Crossentropy(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致). 将根据标签的实际形状自动使用二分类或者多分类损失函数.
double loss::Crossentropy::PyCall(const Eigen::MatrixXd &y_pred,
                                  const Eigen::MatrixXd &y_true) {
    if (y_pred.cols() == 1) {
        BinaryCrossentropy loss = BinaryCrossentropy();
        return loss.PyCall(y_pred, y_true);
    } else {
        CategoricalCrossentropy loss = CategoricalCrossentropy();
        return loss.PyCall(y_pred, y_true);
    }
}

loss::LogLikelihood::LogLikelihood() {
    this->name = "log_likelihood";
}

loss::LogLikelihood::LogLikelihood(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为真实的标签张量, 模型的参数矩阵和属性的参数矩阵.
double loss::LogLikelihood::PyCall(const Eigen::MatrixXd &y_true,
                                   const Eigen::MatrixXd &beta,
                                   const Eigen::MatrixXd &x_hat) {
    Eigen::MatrixXd temp = x_hat * beta;

    Eigen::MatrixXd loss = -y_true.array() * temp.array() + (1 + temp.array().exp()).log();

    return loss.sum();
}

loss::MeanSquaredError::MeanSquaredError() {
    this->name = "mean_squared_error";
}

loss::MeanSquaredError::MeanSquaredError(std::string name) {
    this->name = std::move(name);
}

// 返回损失值, 输入为两个张量(两个向量必须形状一致).
double loss::MeanSquaredError::PyCall(const Eigen::MatrixXd &y_pred,
                                      const Eigen::MatrixXd &y_true) {
    int y_shape = y_true.rows();
    Eigen::MatrixXd temp = y_pred - y_true;
    double loss = pow(temp.sum(), 2) / (2 * y_shape);

    return loss;
}
