//
// losses.cc
// losses
//
// Create by 孙瑞琦 on 2021/1/25.
//
//

#include "losses.h"

namespace losses {
Loss::Loss() {
    this->name = "loss";
}

Loss::Loss(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
template<typename Dtype, typename Matrix>
Dtype Loss::PyCall(const Matrix &y_pred,
                   const Matrix &y_true,
                   const pybind11::args &args,
                   const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError(); // 与Py后端实现相同, 主动抛出异常.
}

BinaryCrossentropy::BinaryCrossentropy() {
    this->name = "binary_crossentropy";
}

BinaryCrossentropy::BinaryCrossentropy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回损失值, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype BinaryCrossentropy::PyCall(const Matrix &y_pred,
                                 const Matrix &y_true,
                                 const pybind11::args &args,
                                 const pybind11::kwargs &kwargs) {
    auto y_shape = (int32)y_true.rows();

    Matrix loss_l = ((y_true.transpose()).adjoint().array() * y_pred.array().log());
    Matrix loss_r = (1 - y_true.array().transpose().array());
    loss_r = loss_r.adjoint().array() * (1 - y_pred.array()).log();

    return -(loss_l.sum() + loss_r.sum()) / y_shape;
}

CategoricalCrossentropy::CategoricalCrossentropy() {
    this->name = "categorical_crossentropy";
}

CategoricalCrossentropy::CategoricalCrossentropy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回损失值, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype CategoricalCrossentropy::PyCall(const Matrix &y_pred,
                                      const Matrix &y_true,
                                      const pybind11::args &args,
                                      const pybind11::kwargs &kwargs) {
    auto y_shape = (int32)y_true.rows();
    Matrix temp = y_pred.array().log();
    Matrix loss = y_true.array() * temp.array();

    return -loss.sum() / y_shape;
}

Crossentropy::Crossentropy() {
    this->name = "crossentropy";
}

Crossentropy::Crossentropy(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回损失值, 输入为两个张量(两个向量必须形状一致). 将根据标签的实际形状自动使用二分类或者多分类损失函数.
template<typename Dtype, typename Matrix>
Dtype Crossentropy::PyCall(const Matrix &y_pred,
                           const Matrix &y_true,
                           const pybind11::args &args,
                           const pybind11::kwargs &kwargs) {
    if (y_pred.cols() == 1) {
        BinaryCrossentropy loss = BinaryCrossentropy();
        return loss.PyCall<Dtype, Matrix>(y_pred, y_true, args, kwargs);
    } else {
        CategoricalCrossentropy loss = CategoricalCrossentropy();
        return loss.PyCall<Dtype, Matrix>(y_pred, y_true, args, kwargs);
    }
}

LogLikelihood::LogLikelihood() {
    this->name = "log_likelihood";
}

LogLikelihood::LogLikelihood(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回损失值, 输入为真实的标签张量, 模型的参数矩阵和属性的参数矩阵.
template<typename Dtype, typename Matrix>
Dtype LogLikelihood::PyCall(const Matrix &y_true,
                            const Matrix &beta,
                            const pybind11::args &args,
                            const pybind11::kwargs &kwargs) {
    Matrix x_hat = pybind11::cast<Matrix>(args[0]);
    Matrix temp = x_hat * beta;

    Matrix loss = -y_true.array() * temp.array() + (1 + temp.array().exp()).log();

    return loss.sum();
}

MeanSquaredError::MeanSquaredError() {
    this->name = "mean_squared_error";
}

MeanSquaredError::MeanSquaredError(std::string name) {
    this->name = std::move(name);
}

// `Dtype` 兼容32位和64位浮点数, `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// 不支持不同位数模板兼容.
// 返回损失值, 输入为两个张量(两个向量必须形状一致).
template<typename Dtype, typename Matrix>
Dtype MeanSquaredError::PyCall(const Matrix &y_pred,
                               const Matrix &y_true,
                               const pybind11::args &args,
                               const pybind11::kwargs &kwargs) {
    auto y_shape = (int32)y_true.rows();
    Matrix temp = y_pred - y_true;
    Dtype loss = pow(temp.sum(), 2) / (2 * y_shape);

    return loss;
}

// 显式实例化.
template float32 Loss::PyCall(const matrix32 &y_pred,
                              const matrix32 &y_true,
                              const pybind11::args &args,
                              const pybind11::kwargs &kwargs);
template float64 Loss::PyCall(const matrix64 &y_pred,
                              const matrix64 &y_true,
                              const pybind11::args &args,
                              const pybind11::kwargs &kwargs);

template float32 BinaryCrossentropy::PyCall(const matrix32 &y_pred,
                                            const matrix32 &y_true,
                                            const pybind11::args &args,
                                            const pybind11::kwargs &kwargs);
template float64 BinaryCrossentropy::PyCall(const matrix64 &y_pred,
                                            const matrix64 &y_true,
                                            const pybind11::args &args,
                                            const pybind11::kwargs &kwargs);

template float32 CategoricalCrossentropy::PyCall(const matrix32 &y_pred,
                                                 const matrix32 &y_true,
                                                 const pybind11::args &args,
                                                 const pybind11::kwargs &kwargs);
template float64 CategoricalCrossentropy::PyCall(const matrix64 &y_pred,
                                                 const matrix64 &y_true,
                                                 const pybind11::args &args,
                                                 const pybind11::kwargs &kwargs);

template float32 Crossentropy::PyCall(const matrix32 &y_pred,
                                      const matrix32 &y_true,
                                      const pybind11::args &args,
                                      const pybind11::kwargs &kwargs);
template float64 Crossentropy::PyCall(const matrix64 &y_pred,
                                      const matrix64 &y_true,
                                      const pybind11::args &args,
                                      const pybind11::kwargs &kwargs);

template float32 LogLikelihood::PyCall(const matrix32 &y_true,
                                       const matrix32 &beta,
                                       const pybind11::args &args,
                                       const pybind11::kwargs &kwargs);
template float64 LogLikelihood::PyCall(const matrix64 &y_true,
                                       const matrix64 &beta,
                                       const pybind11::args &args,
                                       const pybind11::kwargs &kwargs);

template float32 MeanSquaredError::PyCall(const matrix32 &y_pred,
                                          const matrix32 &y_true,
                                          const pybind11::args &args,
                                          const pybind11::kwargs &kwargs);
template float64 MeanSquaredError::PyCall(const matrix64 &y_pred,
                                          const matrix64 &y_true,
                                          const pybind11::args &args,
                                          const pybind11::kwargs &kwargs);
}  // namespace losses
