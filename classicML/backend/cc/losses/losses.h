//
// losses.h
// losses
//
// Create by 孙瑞琦 on 2021/1/25.
//
//

#ifndef CLASSICML_BACKEND_CC_LOSSES_LOSSES_H_
#define CLASSICML_BACKEND_CC_LOSSES_LOSSES_H_

#include "pybind11/eigen.h"

#include "../dtypes.h"
#include "../exceptions.h"

namespace losses {
// 损失函数的基类.
class Loss {
  public:
    Loss();
    explicit Loss(std::string name);
    virtual ~Loss() = default;

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_pred,
                 const Matrix &y_true,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);

  public:
    std::string name;
};

// 二分类交叉熵损失函数.
class BinaryCrossentropy : public Loss {
  public:
    BinaryCrossentropy();
    explicit BinaryCrossentropy(std::string name);

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_pred,
                 const Matrix &y_true,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);
};

// 多分类交叉熵损失函数.
class CategoricalCrossentropy : public Loss {
  public:
    CategoricalCrossentropy();
    explicit CategoricalCrossentropy(std::string name);

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_pred,
                 const Matrix &y_true,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);
};

// 交叉熵损失函数.
class Crossentropy : public Loss {
  public:
    Crossentropy();
    explicit Crossentropy(std::string name);

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_pred,
                 const Matrix &y_true,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);
};

// 对数似然损失函数.
class LogLikelihood : public Loss {
  public:
    LogLikelihood();
    explicit LogLikelihood(std::string name);

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_true,
                 const Matrix &beta,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);
};

// 均方误差损失函数.
class MeanSquaredError : public Loss {
  public:
    MeanSquaredError();
    explicit MeanSquaredError(std::string name);

    template<typename Dtype, typename Matrix>
    Dtype PyCall(const Matrix &y_pred,
                 const Matrix &y_true,
                 const pybind11::args &args,
                 const pybind11::kwargs &kwargs);
};
}  // namespace losses

#endif /* CLASSICML_BACKEND_CC_LOSSES_LOSSES_H_ */