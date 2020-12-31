//
// metrics.h
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#ifndef METRICS_H
#define METRICS_H

#include <utility>

#include "Eigen/Dense"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

class NotImplementedError: public std::exception {
  public:
    const char * what() const noexcept override {
        return "基类没有实现";
    }
};

// 评估函数的基类.
class Metric {
  public:
    Metric();
    explicit Metric(std::string name);

    virtual double PyCall(const Eigen::MatrixXd &y_pred,
                          const Eigen::MatrixXd &y_true);

  public:
    std::string name;
};

// 准确率评估函数.
class Accuracy: public Metric {
public:
    Accuracy();
    explicit Accuracy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

public:
    std::string name;
};

// 二分类准确率评估函数.
class BinaryAccuracy: public Metric {
  public:
    BinaryAccuracy();
    explicit BinaryAccuracy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

// 多分类准确率评估函数.
class CategoricalAccuracy: public Metric {
  public:
    CategoricalAccuracy();
    explicit CategoricalAccuracy(std::string name);

    double PyCall(const Eigen::MatrixXd &y_pred,
                  const Eigen::MatrixXd &y_true) override;

  public:
    std::string name;
};

PYBIND11_MODULE(metrics, m) {
    m.doc() = R"pbdoc(classicML的评估函数, 以CC实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<Metric>(m, "Metric", pybind11::dynamic_attr(), R"pbdoc(
评估函数的基类.

    Attributes:
        name: str, default=None,
            评估函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default=None,
                评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default=None,
                评估函数名称.
)pbdoc",
             pybind11::arg("name"))
        .def_readonly("name", &Metric::name)
        .def("__call__", &Metric::PyCall,
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<Accuracy, Metric>(m, "Accuracy", pybind11::dynamic_attr(), R"pbdoc(
准确率评估函数,
    将根据标签的实际形状自动使用二分类或者多分类评估函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default=None,
                评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default=None,
                评估函数名称.
)pbdoc",
             pybind11::arg("name"))
        .def_readonly("name", &Accuracy::name)
        .def("__call__", &Accuracy::PyCall, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<BinaryAccuracy, Metric>(m, "BinaryAccuracy", pybind11::dynamic_attr(), R"pbdoc(
二分类准确率评估函数.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def_readonly("name", &BinaryAccuracy::name)
        .def("__call__", &BinaryAccuracy::PyCall, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<CategoricalAccuracy, Metric>(m, "CategoricalAccuracy", pybind11::dynamic_attr(), R"pbdoc(
多分类准确率评估函数.
)pbdoc")
            .def(pybind11::init())
            .def(pybind11::init<std::string>(), pybind11::arg("name"))
            .def_readonly("name", &CategoricalAccuracy::name)
            .def("__call__", &CategoricalAccuracy::PyCall, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
                 pybind11::arg("y_pred"), pybind11::arg("y_true"));

    m.attr("__version__") = "backend.cc.metrics.0.2";
}

#endif /* METRICS_H */