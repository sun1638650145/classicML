//
// kernels.h
// kernels
//
// Create by 孙瑞琦 on 2020/2/10.
//
//

#ifndef KERNELS_H
#define KERNELS_H

#include "Eigen/Dense"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "exceptions.h"

// 损失函数的基类.
class Kernel {
  public:
    Kernel();
    explicit Kernel(std::string name);

    virtual Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                                   const Eigen::MatrixXd &x_j);

  public:
    std::string name;
};

// 线性核函数.
class Linear: public Kernel {
  public:
    Linear();
    explicit Linear(std::string name);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                           const Eigen::MatrixXd &x_j) override;

  public:
    std::string name;
};

// 多项式核函数.
class Polynomial: public Kernel {
  public:
    Polynomial();
    explicit Polynomial(std::string name, double gamma, int degree);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                           const Eigen::MatrixXd &x_j) override;

  public:
    std::string name;
    double gamma;
    int degree;
};

// 径向基核函数.
class RBF: public Kernel {
  public:
    RBF();
    explicit RBF(std::string name, double gamma);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                           const Eigen::MatrixXd &x_j) override;

  public:
    std::string name;
    double gamma;
};

// 高斯核函数.
class Gaussian: public RBF {
  public:
    Gaussian();
    explicit Gaussian(std::string name, double gamma);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                           const Eigen::MatrixXd &x_j) override;

 public:
    std::string name;
    double gamma;
};

// Sigmoid核函数.
class Sigmoid: public Kernel {
  public:
    Sigmoid();
    explicit Sigmoid(std::string name, double gamma, double beta, double theta);

    Eigen::MatrixXd PyCall(const Eigen::MatrixXd &x_i,
                           const Eigen::MatrixXd &x_j) override;

  public:
    std::string name;
    double gamma;
    double beta;
    double theta;
};

PYBIND11_MODULE(kernels, m) {
    m.doc() = R"pbdoc(classicML的核函数, 以CC实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<Kernel>(m, "Kernel", pybind11::dynamic_attr(), R"pbdoc(
核函数的基类.

    Attributes:
        name: str, default=None,
            核函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default=None,
                核函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default=None,
                核函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Kernel::name)
        .def("__call__", &Kernel::PyCall, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<Linear, Kernel>(m, "Linear", pybind11::dynamic_attr(), R"pbdoc(
线性核函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='linear',
                核函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='linear',
                核函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &Linear::name)
        .def("__call__", &Linear::PyCall, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<Polynomial, Kernel>(m, "Polynomial", pybind11::dynamic_attr(), R"pbdoc(
多项式核函数.

    Attributes:
        name: str, default='poly',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
        degree: int, default=3,
            多项式的次数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='poly',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
            degree: int, default=3,
                多项式的次数.
)pbdoc")
        .def(pybind11::init<std::string, double, int>(), R"pbdoc(
        Arguments:
            name: str, default='poly',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
            degree: int, default=3,
                多项式的次数.
)pbdoc",
             pybind11::arg("name")="poly", pybind11::arg("gamma")=1.0, pybind11::arg("degree")=3)
        .def_readonly("name", &Polynomial::name)
        .def_readonly("gamma", &Polynomial::gamma)
        .def_readonly("degree", &Polynomial::degree)
        .def("__call__", &Polynomial::PyCall, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<RBF, Kernel>(m, "RBF", pybind11::dynamic_attr(), R"pbdoc(
径向基核函数.

    Attributes:
        name: str, default='rbf',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='rbf',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
)pbdoc")
        .def(pybind11::init<std::string, double>(), R"pbdoc(
        Arguments:
            name: str, default='rbf',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
)pbdoc",
             pybind11::arg("name")="rbf", pybind11::arg("gamma")=1.0)
        .def_readonly("name", &RBF::name)
        .def_readonly("gamma", &RBF::gamma)
        .def("__call__", &RBF::PyCall, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<Gaussian, Kernel>(m, "Gaussian", pybind11::dynamic_attr(), R"pbdoc(
高斯核函数.
    具体实现参看径向基核函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='gaussian',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
)pbdoc")
        .def(pybind11::init<std::string, double>(), R"pbdoc(
        Arguments:
            name: str, default='gaussian',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
)pbdoc",
             pybind11::arg("name")="gaussian", pybind11::arg("gamma")=1.0)
        .def_readonly("name", &Gaussian::name)
        .def_readonly("gamma", &Gaussian::gamma)
        .def("__call__", &Gaussian::PyCall,pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<Sigmoid, Kernel>(m, "Sigmoid", pybind11::dynamic_attr(), R"pbdoc(
Sigmoid核函数.

    Attributes:
        name: str, default='sigmoid',
            核函数名称.
        gamma: float, default=1.0,
            核函数系数.
        beta: float, default=1.0,
            核函数参数.
        theta: float, default=-1.0,
            核函数参数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='sigmoid',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
            beta: float, default=1.0,
                核函数参数.
            theta: float, default=-1.0,
                核函数参数.
)pbdoc")
        .def(pybind11::init<std::string, double, double, double>(), R"pbdoc(
        Arguments:
            name: str, default='sigmoid',
                核函数名称.
            gamma: float, default=1.0,
                核函数系数.
            beta: float, default=1.0,
                核函数参数.
            theta: float, default=-1.0,
                核函数参数.
)pbdoc",
             pybind11::arg("name")="sigmoid", pybind11::arg("gamma")=1.0,
             pybind11::arg("beta")=1.0, pybind11::arg("theta")=-1.0)
        .def_readonly("name", &Sigmoid::name)
        .def_readonly("gamma", &Sigmoid::gamma)
        .def_readonly("beta", &Sigmoid::beta)
        .def_readonly("theta", &Sigmoid::theta)
        .def("__call__", &Sigmoid::PyCall, pybind11::arg("x_i"), pybind11::arg("x_j"));

    m.attr("__version__") = "backend.cc.kernels.0.3";
}

#endif /* KERNELS_H */