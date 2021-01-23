//
// activations.h
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

#include "Eigen/Dense"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "exceptions.h"

// 激活函数基类.
class Activation {
  public:
    Activation();
    explicit Activation(std::string name);

    virtual Eigen::MatrixXd PyCall(const Eigen::MatrixXd &z);
    virtual Eigen::MatrixXd Diff(const Eigen::MatrixXd &output,
                                 const Eigen::MatrixXd &a);

  public:
    std::string name;
};

// ReLU激活函数.
class Relu: public Activation {
  public:
    Relu();
    explicit Relu(std::string name);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &z) override;
    Eigen::MatrixXd Diff(const Eigen::MatrixXd &output,
                         const Eigen::MatrixXd &a) override;

  public:
    std::string name;
};

// Sigmoid激活函数.
class Sigmoid: public Activation {
  public:
    Sigmoid();
    explicit Sigmoid(std::string name);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &z) override;
    Eigen::MatrixXd Diff(const Eigen::MatrixXd &output,
                         const Eigen::MatrixXd &a,
                         const Eigen::MatrixXd &y_true);

  public:
    std::string name;
};

// Softmax激活函数.
class Softmax: public Activation {
  public:
    Softmax();
    explicit Softmax(std::string name);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &z) override;
    Eigen::MatrixXd Diff(const Eigen::MatrixXd &output,
                         const Eigen::MatrixXd &a) override;

  public:
    std::string name;
};

PYBIND11_MODULE(activations, m) {
    m.doc() = R"pbdoc(classicML的激活函数, 以CC实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<Activation>(m, "Activation", pybind11::dynamic_attr(), R"pbdoc(
激活函数基类.

    Attributes:
        name: str, default='activations',
            激活函数名称.

    Raises:
       NotImplementedError: __call__, diff方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='activations',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='activations',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Activation::name)
        .def("__call__", &Activation::PyCall, pybind11::arg("z"))
        .def("diff", &Activation::Diff, pybind11::arg("output"), pybind11::arg("a"));

    pybind11::class_<Relu, Activation>(m, "Relu", pybind11::dynamic_attr(), R"pbdoc(
ReLU激活函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='activations',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='activations',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Relu::name)
        .def("__call__", &Relu::PyCall, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &Relu::Diff, R"pbdoc(
ReLU函数的微分.

    Arguments:
        output: numpy.ndarray, 前向传播输出的张量.
        a: numpy.ndarray, 输入的张量.

    Notes:
        ReLU函数在大于零区间的导数应该是恒为一, 如果按此计算在实际应用上会随着训练轮数的增加, 最后模型的输出是一个随机概率,
        作者个人认为原因是随着轮数的增加, 大部分神经元都恒为激活态, 成为一个线性操作(缩放实际上相当于不参与计算了).
        在实际应用中使用原值发现可以避免这种想象.
)pbdoc",pybind11::arg("output"), pybind11::arg("a"));

    pybind11::class_<Sigmoid, Activation>(m, "Sigmoid", pybind11::dynamic_attr(), R"pbdoc(
Sigmoid激活函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='sigmoid',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='sigmoid',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Sigmoid::name)
        .def("__call__", &Sigmoid::PyCall, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &Sigmoid::Diff, R"pbdoc(
ReLU函数的微分.

    Arguments:
        output: numpy.ndarray, 前向传播输出的张量.
        a: numpy.ndarray, 输入的张量.
        args:
            y_true: numpy.ndarray, 真实的标签.

    Returns:
        Sigmoid的导数(微分).

    Notes:
        Sigmoid的导数f' = a * (1 - a),
        但是作为输出层就需要乘上误差.
)pbdoc",pybind11::arg("output"), pybind11::arg("a"), pybind11::arg("y_pred"));

    pybind11::class_<Softmax, Activation>(m, "Softmax", pybind11::dynamic_attr(), R"pbdoc(
Softmax激活函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='softmax',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='softmax',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Softmax::name)
        .def("__call__", &Softmax::PyCall, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &Softmax::Diff, R"pbdoc(
Sigmoid的导数(微分).
    Arguments:
        output: numpy.ndarray, 输出张量.
        a: numpy.ndarray, 输入张量.
        args:
            y_true: numpy.ndarray, 真实的标签.

    Returns:
        Sigmoid的导数(微分).

    Notes:
        Sigmoid的导数f' = a * (1 - a),
        但是作为输出层就需要乘上误差.
)pbdoc",pybind11::arg("output"), pybind11::arg("a"));

    // Instances.
    m.attr("relu") = Relu();
    m.attr("sigmoid") = Sigmoid();
    m.attr("softmax") = Softmax();

    m.attr("__version__") = "backend.cc.activations.0.3";
}

#endif /* ACTIVATIONS_H */