//
// losses.h
// losses
//
// Create by 孙瑞琦 on 2020/1/25.
//
//

#ifndef LOSSES_H
#define LOSSES_H

#include <cmath>

#include "Eigen/Dense"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "exceptions.h"

// 损失函数的基类.
class Loss {
  public:
    Loss();
    explicit Loss(std::string name);

    virtual double PyCall(const Eigen::MatrixXd &y_pred,
                          const Eigen::MatrixXd &y_true);

  public:
    std::string name;
};

// 二分类交叉熵损失函数.
class BinaryCrossentropy: public Loss {
  public:
    BinaryCrossentropy();
    explicit BinaryCrossentropy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

// 多分类交叉熵损失函数.
class CategoricalCrossentropy: public Loss {
  public:
    CategoricalCrossentropy();
    explicit CategoricalCrossentropy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

// 交叉熵损失函数.
class Crossentropy: public Loss {
  public:
    Crossentropy();
    explicit Crossentropy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

// 对数似然损失函数.
class LogLikelihood: public Loss {
  public:
    LogLikelihood();
    explicit LogLikelihood(std::string name);

    double PyCall(const Eigen::MatrixXd &y_true,
                  const Eigen::MatrixXd &beta,
                  const Eigen::MatrixXd &x_hat);

  public:
    std::string name;
};

// 均方误差损失函数.
class MeanSquaredError: public Loss {
  public:
    MeanSquaredError();
    explicit MeanSquaredError(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

// Aliases.
// 受限于CPP语言特性, 必须定义一个子类, 而不能直接使用别名.
class MSE: public MeanSquaredError {
public:
    MSE();
    explicit MSE(std::string name);
};

PYBIND11_MODULE(losses, m) {
    m.doc() = R"pbdoc(classicML的损失函数, 以CC实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<Loss>(m, "Loss", pybind11::dynamic_attr(), R"pbdoc(
损失函数的基类.

    Attributes:
        name: str, default=None,
            损失函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='loss',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default=None,
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Loss::name)
        .def("__call__", &Loss::PyCall, pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<BinaryCrossentropy, Loss>(m, "BinaryCrossentropy", pybind11::dynamic_attr(), R"pbdoc(
二分类交叉熵损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='binary_crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='binary_crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &BinaryCrossentropy::name)
        .def("__call__", &BinaryCrossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<CategoricalCrossentropy, Loss>(m, "CategoricalCrossentropy", pybind11::dynamic_attr(), R"pbdoc(
多分类交叉熵损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='categorical_crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='categorical_crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &CategoricalCrossentropy::name)
        .def("__call__", &CategoricalCrossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<Crossentropy, Loss>(m, "Crossentropy", pybind11::dynamic_attr(), R"pbdoc(
交叉熵损失函数,
将根据标签的实际形状自动使用二分类或者多分类损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Crossentropy::name)
        .def("__call__", &Crossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<LogLikelihood, Loss>(m, "LogLikelihood", pybind11::dynamic_attr(), R"pbdoc(
对数似然损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='log_likelihood',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='log_likelihood',
                损失函数名称.
)pbdoc")
        .def_readonly("name", &LogLikelihood::name)
        .def("__call__", &LogLikelihood::PyCall,
             pybind11::arg("y_true"),
             pybind11::arg("beta"),
             pybind11::arg("x_hat"));

    pybind11::class_<MeanSquaredError, Loss>(m, "MeanSquaredError", pybind11::dynamic_attr(), R"pbdoc(
均方误差损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def_readonly("name", &MeanSquaredError::name)
        .def("__call__", &MeanSquaredError::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    // Aliases.
    pybind11::class_<MSE, Loss>(m, "MSE", pybind11::dynamic_attr(), R"pbdoc(
均方误差损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def_readonly("name", &MSE::name);

    m.attr("__version__") = "backend.cc.losses.0.4";
}

#endif /* LOSSES_H */