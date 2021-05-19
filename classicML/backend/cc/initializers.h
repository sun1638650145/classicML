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
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "exceptions.h"
#include "matrix_op.h"

namespace initializers {
// 初始化器的基类.
class Initializer {
    public:
        Initializer();
        explicit Initializer(std::string name);
        explicit Initializer(std::string name, std::optional<unsigned int> seed);

        virtual Eigen::MatrixXd PyCall(const pybind11::args &args,
                                       const pybind11::kwargs &kwargs);

    public:
        std::string name;
        std::optional<unsigned int> seed;
};

// 正态分布随机初始化器.
class RandomNormal: public Initializer {
    public:
        RandomNormal();
        explicit RandomNormal(std::string name);
        explicit RandomNormal(std::string name, std::optional<unsigned int> seed);

        // overload
        Eigen::MatrixXd PyCall(const int &attributes_or_structure);
        std::map<std::string, Eigen::MatrixXd> PyCall(const Eigen::RowVectorXi &attributes_or_structure);

    public:
        std::string name;
        std::optional<unsigned int> seed;
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
        .def(pybind11::init<std::string, std::optional<unsigned int>>(), R"pbdoc(
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

    pybind11::class_<initializers::RandomNormal, initializers::Initializer>(m, "RandomNormal", pybind11::dynamic_attr(),
R"pbdoc(
正态分布随机初始化器.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="random_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readonly("name", &initializers::RandomNormal::name)
        .def_readonly("seed", &initializers::RandomNormal::seed)
        .def("__call__", pybind11::overload_cast<const int &>(&initializers::RandomNormal::PyCall),
R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", pybind11::overload_cast<const Eigen::RowVectorXi &>(&initializers::RandomNormal::PyCall),
R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"));

    m.attr("__version__") = "backend.cc.initializers.0.2";
}
#endif /* INITIALIZERS_H */