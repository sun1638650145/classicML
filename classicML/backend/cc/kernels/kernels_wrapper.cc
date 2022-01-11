//
// kernels_wrapper.cc
// kernels
//
// Created by 孙瑞琦 on 2021/12/20.
//
//

#include "pybind11/pybind11.h"

#include "kernels.h"

PYBIND11_MODULE(kernels, m) {
    m.doc() = R"pbdoc(classicML的核函数, 以C++实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<kernels::Kernel>(m, "Kernel", R"pbdoc(
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
        .def_readwrite("name", &kernels::Kernel::name)
        .def("__call__", &kernels::Kernel::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::Kernel::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<kernels::Linear, kernels::Kernel>(m, "Linear", R"pbdoc(
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
        .def_readwrite("name", &kernels::Linear::name)
        .def("__call__", &kernels::Linear::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::Linear::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<kernels::Polynomial, kernels::Kernel>(m, "Polynomial", R"pbdoc(
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
        .def(pybind11::init<std::string, float64, int32>(), R"pbdoc(
Arguments:
    name: str, default='poly',
        核函数名称.
    gamma: float, default=1.0,
        核函数系数.
    degree: int, default=3,
        多项式的次数.
)pbdoc", pybind11::arg("name")="poly", pybind11::arg("gamma")=1.0, pybind11::arg("degree")=3)
        .def_readwrite("name", &kernels::Polynomial::name)
        .def_readwrite("gamma", &kernels::Polynomial::gamma)
        .def_readwrite("degree", &kernels::Polynomial::degree)
        .def("__call__", &kernels::Polynomial::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::Polynomial::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<kernels::RBF, kernels::Kernel>(m, "RBF", R"pbdoc(
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
        .def(pybind11::init<std::string, float64>(), R"pbdoc(
Arguments:
    name: str, default='rbf',
        核函数名称.
    gamma: float, default=1.0,
        核函数系数.
)pbdoc", pybind11::arg("name")="rbf", pybind11::arg("gamma")=1.0)
        .def_readwrite("name", &kernels::RBF::name)
        .def_readwrite("gamma", &kernels::RBF::gamma)
        .def("__call__", &kernels::RBF::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::RBF::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<kernels::Gaussian, kernels::RBF>(m, "Gaussian", R"pbdoc(
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
        .def(pybind11::init<std::string, float64>(), R"pbdoc(
Arguments:
    name: str, default='gaussian',
        核函数名称.
    gamma: float, default=1.0,
        核函数系数.
)pbdoc", pybind11::arg("name")="gaussian", pybind11::arg("gamma")=1.0)
        .def_readwrite("name", &kernels::Gaussian::name)
        .def_readwrite("gamma", &kernels::Gaussian::gamma)
        .def("__call__", &kernels::Gaussian::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::Gaussian::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    pybind11::class_<kernels::Sigmoid, kernels::Kernel>(m, "Sigmoid", R"pbdoc(
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
        .def(pybind11::init<std::string, float64, float64, float64>(), R"pbdoc(
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
            pybind11::arg("name")="sigmoid",
            pybind11::arg("gamma")=1.0,
            pybind11::arg("beta")=1.0,
            pybind11::arg("theta")=-1.0)
        .def_readwrite("name", &kernels::Sigmoid::name)
        .def_readwrite("gamma", &kernels::Sigmoid::gamma)
        .def_readwrite("beta", &kernels::Sigmoid::beta)
        .def_readwrite("theta", &kernels::Sigmoid::theta)
        .def("__call__", &kernels::Sigmoid::PyCall<matrix32>, pybind11::arg("x_i"), pybind11::arg("x_j"))
        .def("__call__", &kernels::Sigmoid::PyCall<matrix64>, pybind11::arg("x_i"), pybind11::arg("x_j"));

    m.attr("__version__") = "backend.cc.kernels.0.10.3";
}