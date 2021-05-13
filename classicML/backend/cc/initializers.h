//
// initializers.h
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
//
//

#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "Eigen/Core"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "exceptions.h"

namespace initializers {
// 初始化器的基类.
class Initializer {
    public:
        Initializer();
        explicit Initializer(std::string name);
        explicit Initializer(std::string name, int seed);

        virtual Eigen::MatrixXd PyCall(const pybind11::args &args,
                                       const pybind11::kwargs &kwargs);

    public:
        std::string name;
        std::optional<int> seed;
};
}  // namespace initializers

PYBIND11_MODULE(initializers, m) {
    m.doc() = R"pbdoc(classicML的初始化器, 以C++实现)pbdoc";

    // 注册自定义异常.
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<initializers::Initializer>(m, "Initializer", pybind11::dynamic_attr(), R"pbdoc(
初始化器的基类.

    Attributes:
        name: str, default='initializer',
            初始化器的名称.
        seed: int, default=None,
            初始化器的随机种子.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
Arguments:
    name: str, default='initializer',
        初始化器的名称.
    seed: int, default=None,
        初始化器的随机种子.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
Arguments:
    name: str, default='initializer',
        初始化器的名称.
    seed: int, default=None,
        初始化器的随机种子.
)pbdoc",
             pybind11::arg("name"))
        .def(pybind11::init<std::string, int>(), R"pbdoc(
Arguments:
    name: str, default='initializer',
        初始化器的名称.
    seed: int, default=None,
        初始化器的随机种子.
)pbdoc",
             pybind11::arg("name")="initializer",
             pybind11::arg("seed")=pybind11::none())
        .def_readonly("name", &initializers::Initializer::name)
        .def_readonly("seed", &initializers::Initializer::seed)
        .def("__call__", &initializers::Initializer::PyCall);

    m.attr("__version__") = "backend.cc.initializers.0.1";
}
#endif /* INITIALIZERS_H */