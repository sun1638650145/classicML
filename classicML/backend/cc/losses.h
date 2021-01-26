//
// losses.h
// losses
//
// Create by 孙瑞琦 on 2020/1/25.
//
//

#ifndef LOSSES_H
#define LOSSES_H

#include <iostream>

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
            name: str, default=None,
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
            name: str, default=None,
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default=None,
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &BinaryCrossentropy::name)
        .def("__call__", &BinaryCrossentropy::PyCall, pybind11::arg("y_pred"), pybind11::arg("y_true"));

    m.attr("__version__") = "backend.cc.losses.0.1";
}
#endif /* LOSSES_H */