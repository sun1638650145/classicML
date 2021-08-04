//
// initializers.h
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
//
//

#ifndef CLASSICML_BACKEND_CC_INITIALIZERS_H_
#define CLASSICML_BACKEND_CC_INITIALIZERS_H_

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dtypes.h"
#include "exceptions.h"
#include "matrix_op.h"

namespace initializers {
// 初始化器的基类.
class Initializer {
    public:
        Initializer();
        explicit Initializer(std::string name);
        explicit Initializer(std::string name, std::optional<uint32> seed);
        virtual ~Initializer() = default;

        // TODO(Steve R. Sun, tag:code): 在C++侧不能使用virtual关键字标记, 编译器会有警告, 但是需要声明此方法子类必须实现.
        template<typename Matrix>
        Matrix PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs);

    public:
        std::string name;
        std::optional<uint32> seed;
};

// 正态分布随机初始化器.
class RandomNormal: public Initializer {
    public:
        RandomNormal();
        explicit RandomNormal(std::string name);
        explicit RandomNormal(std::string name, std::optional<uint32> seed);

        /* 总结(Steve Sun, 2021-07-13, 受限于作者的思想和技术局限性, 在当前时空观下,
         结合《设计心理学第二卷》和《Google开源代码风格指南》的设计理念, 这似乎是最优解,
         记录下面这些在方便理解源码的同时希望来自未来的我本人或者其他开发者能找到更优解):

        0. 鸣谢:
         @Jan Iwaszkiewicz(Intel) 提供了所有的技术解决方案, 包括可行的和不可行的.
         @leishi(Google) 介绍我进了"组织", 并赠予本人CPython 3.9的教材一本.
         @Xiang Zhang(Huawei, 大陆唯一的Python核心贡献者) 让我下定决心放弃了使用template的想法,
            使用template极大影响可阅读性, 同时并不pythonic.

        1. 特别鸣谢: @邵常龙

        2.版本:
         c++std: c++17
         pybind11: 2.6.2
         eigen: 3.3.9
         numpy: 1.20.3
         macOS: 11.4 (with Apple Silicon M1 16GB RAM)
         运行环境:
         Python: 3.8.10(Clang 11.1.0|conda 4.10.1)
         编译环境:
         Clang: Apple clang version 12.0.5 (clang-1205.0.22.11)
         Target: arm64-apple-darwin20.5.0

        3.关于使用template的问题: 使用template后极大的降低代码的可阅读性,
            全局引入template在函数体内仍旧需要使用4个分支, 不能减少代码量, 故pass;
            折衷方案仅在返回字典上使用了template.

            问题参考: [I have two overloaded functions, one of which uses a template,
            how should I convert to the Python side?](https://github.com/pybind/pybind11/issues/3085)

        4.关于数据类型: Pure Python 3.x无法自动区分32位和64位整型, 需要引入Numpy的数据类型.
            4.1.引入Numpy的数据类型后, pybind11总是会转换为int, 导致Python Interpreter无法正确区分32位和64位整型.
            解决上述问题, 使用了一个trick, 利用了Python C API 直接访问缓冲区.

            问题参考: [How to bind NumPy scalar?](https://github.com/pybind/pybind11/issues/3093)
                     [NumPy scalars](https://github.com/pybind/pybind11/pull/2060)
                     此PR已于两年前merge失败, 同时本地使用不能通过软件测试, 故pass.
                     [Efficient arrays of numeric values](https://docs.python.org/3.8/library/array.html)

            4.2.Python Interpreter不能自动解决np.int和Pure int的兼容性,
                Pure int会匹配到np.int32和np.int64, 但是np.int32无法匹配int;
            (但是list和np.ndarray就能, 底层涉及到技术包括CPython, Numpy API和 pybind11; 过于复杂, 深入研究会超出当前的范围.)
            技术上解决, overload(Pure int).

        5.关于Overload匹配顺序: 对于C++来说是匹配的更精准更好.
            实际情况解释器总是匹配顺序: Python的缓冲区(pybind11::buffer) > Eigen::Matrix(numpy.ndarray);
                                    Pure int > np.int;
            5.1.关于Python两点问题:
                5.1.1. Python Interpreter是C实现, 严格说应该会有少量和C++的差异.
                5.1.2. 不针对科学计算Python几乎可以无视重载这个概念.
                5.1.3. Python 似乎总是优先匹配内置的数据类型, 且Python缓冲区的权限最高.

            解决上述问题所使用的重载函数注册顺序:
            `Eigen::RowVectorXi` = `Eigen::Matrix<std::int64_t, 1, -1>` > `pybind11::buffer(内含32和64位的自动切换)` > Pure int.
         */

        // overload
        template<typename Matrix, typename RowVector, typename Dtype>
        std::map<std::string, Matrix> PyCall(const RowVector &attributes_or_structure);

        std::variant<Eigen::MatrixXf, Eigen::MatrixXd> PyCall(const pybind11::buffer &attributes_or_structure);
        Eigen::MatrixXf PyCall(const int32 &attributes_or_structure);
};

// He正态分布随机初始化器.
class HeNormal: public Initializer {
    public:
        HeNormal();
        explicit HeNormal(std::string name);
        explicit HeNormal(std::string name, std::optional<uint32> seed);

        // overload
        template<typename Matrix, typename RowVector, typename Dtype>
        std::map<std::string, Matrix> PyCall(const RowVector &attributes_or_structure);

        std::variant<Eigen::MatrixXf, Eigen::MatrixXd> PyCall(const pybind11::buffer &attributes_or_structure);
        Eigen::MatrixXf PyCall(const int32 &attributes_or_structure);
};

// Xavier正态分布随机初始化器.
class XavierNormal: public Initializer {
    public:
        XavierNormal();
        explicit XavierNormal(std::string name);
        explicit XavierNormal(std::string name, std::optional<uint32> seed);

        // overload
        template<typename Matrix, typename RowVector, typename Dtype>
        std::map<std::string, Matrix> PyCall(const RowVector &attributes_or_structure);

        std::variant<Eigen::MatrixXf, Eigen::MatrixXd> PyCall(const pybind11::buffer &attributes_or_structure);
        Eigen::MatrixXf PyCall(const int32 &attributes_or_structure);
};

// Glorot正态分布随机初始化器.
class GlorotNormal: public XavierNormal {
    public:
        GlorotNormal();
        explicit GlorotNormal(std::string name);
        explicit GlorotNormal(std::string name, std::optional<uint32> seed);
};

// RBF网络的初始化器.
class RBFNormal: public Initializer {
    public:
        RBFNormal();
        explicit RBFNormal(std::string name);
        explicit RBFNormal(std::string name, std::optional<uint32> seed);

        // overload
        std::variant<std::map<std::string, Eigen::MatrixXf>, std::map<std::string, Eigen::MatrixXd>>
        PyCall(const pybind11::buffer &hidden_units);

        std::map<std::string, Eigen::MatrixXf> PyCall(const int32 &hidden_units);
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
        .def(pybind11::init<std::string, std::optional<uint32>>(), R"pbdoc(
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
        .def("__call__", &initializers::Initializer::PyCall<Eigen::MatrixXf>)
        .def("__call__", &initializers::Initializer::PyCall<Eigen::MatrixXd>);

    pybind11::class_<initializers::RandomNormal, initializers::Initializer>(m, "RandomNormal", R"pbdoc(
正态分布随机初始化器.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<uint32>>(),
             pybind11::arg("name")="random_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::RandomNormal::name)
        .def_readwrite("seed", &initializers::RandomNormal::seed)
        .def("__call__", [](initializers::RandomNormal &self, const Eigen::RowVectorXi &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXf, Eigen::RowVectorXi, float32>(attributes_or_structure);
        }, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::RandomNormal &self, const Eigen::Matrix<int64, 1, -1> &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXd, Eigen::Matrix<int64, 1, -1>, float64>(attributes_or_structure);
        }, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::RandomNormal &self, const pybind11::buffer &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::RandomNormal &self, const int32 &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
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
        .def(pybind11::init<std::string, std::optional<uint32>>(),
             pybind11::arg("name")="he_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::HeNormal::name)
        .def_readwrite("seed", &initializers::HeNormal::seed)
        .def("__call__", [](initializers::HeNormal &self, const Eigen::RowVectorXi &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXf, Eigen::RowVectorXi, float32>(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::HeNormal &self, const Eigen::Matrix<int64, 1, -1> &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXd, Eigen::Matrix<int64, 1, -1>, float64>(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::HeNormal &self, const pybind11::buffer &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::HeNormal &self, const int32 &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
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
        .def(pybind11::init<std::string, std::optional<uint32>>(),
             pybind11::arg("name")="xavier_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::XavierNormal::name)
        .def_readwrite("seed", &initializers::XavierNormal::seed)
        .def("__call__", [](initializers::XavierNormal &self, const Eigen::RowVectorXi &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXf, Eigen::RowVectorXi, float32>(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::XavierNormal &self, const Eigen::Matrix<int64, 1, -1> &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXd, Eigen::Matrix<int64, 1, -1>, float64>(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::XavierNormal &self, const pybind11::buffer &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::XavierNormal &self, const int32 &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, R"pbdoc(
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
        .def(pybind11::init<std::string, std::optional<uint32>>(),
             pybind11::arg("name")="glorot_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::GlorotNormal::name)
        .def_readwrite("seed", &initializers::GlorotNormal::seed)
        .def("__call__", [](initializers::GlorotNormal &self, const Eigen::RowVectorXi &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXf, Eigen::RowVectorXi, float32>(attributes_or_structure);
        }, pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::GlorotNormal &self, const Eigen::Matrix<int64, 1, -1> &attributes_or_structure) {
            return self.PyCall<Eigen::MatrixXd, Eigen::Matrix<int64, 1, -1>, float64>(attributes_or_structure);
        }, pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::GlorotNormal &self, const pybind11::buffer &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, pybind11::arg("attributes_or_structure"))
        .def("__call__", [](initializers::GlorotNormal &self, const int32 &attributes_or_structure) {
            return self.PyCall(attributes_or_structure);
        }, pybind11::arg("attributes_or_structure"));

    pybind11::class_<initializers::RBFNormal, initializers::Initializer>(m, "RBFNormal", R"pbdoc(
RBF网络的初始化器.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<uint32>>(),
             pybind11::arg("name")="rbf_normal",
             pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::RBFNormal::name)
        .def_readwrite("seed", &initializers::RBFNormal::seed)
        .def("__call__", pybind11::overload_cast<const pybind11::buffer &>(&initializers::RBFNormal::PyCall), R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"))
        .def("__call__", pybind11::overload_cast<const int32 &>(&initializers::RBFNormal::PyCall), R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"));

    m.attr("__version__") = "backend.cc.initializers.0.5.b2";
}
#endif /* CLASSICML_BACKEND_CC_INITIALIZERS_H_ */