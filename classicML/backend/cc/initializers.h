//
// initializers.h
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
//
//

#ifndef CLASSICML_BACKEND_CC_INITIALIZERS_H_
#define CLASSICML_BACKEND_CC_INITIALIZERS_H_

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
        virtual ~Initializer() = default;

        // TODO(Steve R. Sun tag:code): 在C++侧不能使用virtual关键字标记, 编译器会有警告, 但是需要声明此方法子类必须实现.
        Eigen::MatrixXd PyCall(const pybind11::args &args,
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
};

// He正态分布随机初始化器.
class HeNormal: public Initializer {
    public:
        HeNormal();
        explicit HeNormal(std::string name);
        explicit HeNormal(std::string name, std::optional<unsigned int> seed);

        // overload
        Eigen::MatrixXd PyCall(const int &attributes_or_structure);
        std::map<std::string, Eigen::MatrixXd> PyCall(const Eigen::RowVectorXi &attributes_or_structure);
};

// Xavier正态分布随机初始化器.
class XavierNormal: public Initializer {
    public:
        XavierNormal();
        explicit XavierNormal(std::string name);
        explicit XavierNormal(std::string name, std::optional<unsigned int> seed);

        // overload
        Eigen::MatrixXd PyCall(const int &attributes_or_structure);
        std::map<std::string, Eigen::MatrixXd> PyCall(const Eigen::RowVectorXi &attributes_or_structure);
};

// Glorot正态分布随机初始化器.
class GlorotNormal: public XavierNormal {
    public:
        GlorotNormal();
        explicit GlorotNormal(std::string name);
        explicit GlorotNormal(std::string name, std::optional<unsigned int> seed);
};

// RBF网络的初始化器.
class RBFNormal: public Initializer {
    public:
        RBFNormal();
        explicit RBFNormal(std::string name);
        explicit RBFNormal(std::string name, std::optional<unsigned int> seed);

        // overload
        std::map<std::string, Eigen::MatrixXd> PyCall(const int &hidden_units);
};
}  // namespace initializers

PYBIND11_MODULE(initializers, m) {
    m.doc() = R"pbdoc(classicML的初始化器, 以C++实现)pbdoc";

    // 注册自定义异常.
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<initializers::Initializer>(m, "Initializer", R"pbdoc(
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
        .def_readwrite("name", &initializers::Initializer::name)
        .def_readwrite("seed", &initializers::Initializer::seed)
        .def("__call__", &initializers::Initializer::PyCall);

    pybind11::class_<initializers::RandomNormal, initializers::Initializer>(m, "RandomNormal", R"pbdoc(
正态分布随机初始化器.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="random_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::RandomNormal::name)
        .def_readwrite("seed", &initializers::RandomNormal::seed)
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

    pybind11::class_<initializers::HeNormal, initializers::Initializer>(m, "HeNormal", R"pbdoc(
He正态分布随机初始化器.

    References:
        - [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
          ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="he_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::HeNormal::name)
        .def_readwrite("seed", &initializers::HeNormal::seed)
        .def("__call__", pybind11::overload_cast<const int &>(&initializers::HeNormal::PyCall),
R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", pybind11::overload_cast<const Eigen::RowVectorXi &>(&initializers::HeNormal::PyCall),
R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"));

    pybind11::class_<initializers::XavierNormal, initializers::Initializer>(m, "XavierNormal", R"pbdoc(
Xavier正态分布随机初始化器,
        也叫做Glorot正态分布随机初始化器.

    References:
        - [Glorot et al., 2010](https://proceedings.mlr.press/v9/glorot10a.html)
          ([pdf](https://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="xavier_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::XavierNormal::name)
        .def_readwrite("seed", &initializers::XavierNormal::seed)
        .def("__call__", pybind11::overload_cast<const int &>(&initializers::XavierNormal::PyCall),
R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", pybind11::overload_cast<const Eigen::RowVectorXi &>(&initializers::XavierNormal::PyCall),
R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"));

    pybind11::class_<initializers::GlorotNormal, initializers::XavierNormal>(m, "GlorotNormal", R"pbdoc(
Glorot正态分布随机初始化器.
 具体实现参看XavierNormal.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="glorot_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::GlorotNormal::name)
        .def_readwrite("seed", &initializers::GlorotNormal::seed)
        .def("__call__", pybind11::overload_cast<const int &>(&initializers::GlorotNormal::PyCall),
             pybind11::arg("attributes_or_structure"))
        .def("__call__", pybind11::overload_cast<const Eigen::RowVectorXi &>(&initializers::GlorotNormal::PyCall),
             pybind11::arg("attributes_or_structure"));

    pybind11::class_<initializers::RBFNormal, initializers::Initializer>(m, "RBFNormal", R"pbdoc(
RBF网络的初始化器.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<unsigned int>>(),
             pybind11::arg("name")="rbf_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::RBFNormal::name)
        .def_readwrite("seed", &initializers::RBFNormal::seed)
        .def("__call__", pybind11::overload_cast<const int &>(&initializers::RBFNormal::PyCall),
R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"));

    m.attr("__version__") = "backend.cc.initializers.0.4.3";
}
#endif /* CLASSICML_BACKEND_CC_INITIALIZERS_H_ */