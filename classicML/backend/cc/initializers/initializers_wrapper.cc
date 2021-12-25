//
// initializers_wrapper.cc
// initializers
//
// Created by 孙瑞琦 on 2021/12/21.
//
//

#include "pybind11/pybind11.h"

#include "initializers.h"

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
)pbdoc", pybind11::arg("name"))
        .def(pybind11::init<std::string, std::optional<uint32>>(), R"pbdoc(
Arguments:
    name: str, default='initializer',
        初始化器的名称.
    seed: int, default=None,
        初始化器的随机种子.
)pbdoc", pybind11::arg("name")="initializer", pybind11::arg("seed")=pybind11::none())
        .def_readwrite("name", &initializers::Initializer::name)
        .def_readwrite("seed", &initializers::Initializer::seed)
        .def("__call__", &initializers::Initializer::PyCall<matrix32>)
        .def("__call__", &initializers::Initializer::PyCall<matrix64>);

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
        /* overload匹配顺序: Pure int兼容np.int32和np.int64, 因此重载函数注册时, np.int32和np.int64必须在int32之前. */
        .def("__call__", &initializers::RandomNormal::PyCall1<matrix32, np_int32, float32>, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::RandomNormal::PyCall1<matrix64, np_int64, float64>, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::RandomNormal::PyCall1<matrix32, int32, float32>, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::RandomNormal::PyCall2<matrix32, row_vector32i, float32>, R"pbdoc(
函数实现.
    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::RandomNormal::PyCall2<matrix64, row_vector64i, float64>, R"pbdoc(
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
        .def("__call__", &initializers::HeNormal::PyCall1<matrix32, np_int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::HeNormal::PyCall1<matrix64, np_int64, float64>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::HeNormal::PyCall1<matrix32, int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::HeNormal::PyCall2<matrix32, row_vector32i, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in)), 其中N_in为对应连接的输入层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::HeNormal::PyCall2<matrix64, row_vector64i, float64>, R"pbdoc(
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
        .def("__call__", &initializers::XavierNormal::PyCall1<matrix32, np_int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

Arguments:
    attributes_or_structure: int or list,
        如果是逻辑回归就是样本的特征数;
        如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::XavierNormal::PyCall1<matrix64, np_int64, float64>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

Arguments:
    attributes_or_structure: int or list,
        如果是逻辑回归就是样本的特征数;
        如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::XavierNormal::PyCall1<matrix32, int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::XavierNormal::PyCall2<matrix32, row_vector32i, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::XavierNormal::PyCall2<matrix64, row_vector64i, float64>, R"pbdoc(
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
        .def("__call__", &initializers::GlorotNormal::PyCall1<matrix32, np_int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

Arguments:
    attributes_or_structure: int or list,
        如果是逻辑回归就是样本的特征数;
        如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::GlorotNormal::PyCall1<matrix64, np_int64, float64>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

Arguments:
    attributes_or_structure: int or list,
        如果是逻辑回归就是样本的特征数;
        如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::GlorotNormal::PyCall1<matrix32, int32, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::GlorotNormal::PyCall2<matrix32, row_vector32i, float32>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"))
        .def("__call__", &initializers::GlorotNormal::PyCall2<matrix64, row_vector64i, float64>, R"pbdoc(
初始化方式为W~N(0, sqrt(2/N_in+N_out)),
其中N_in为对应连接的输入层的神经元个数, N_out为本层的神经元个数.

    Arguments:
        attributes_or_structure: int or list,
            如果是逻辑回归就是样本的特征数;
            如果是神经网络, 就是定义神经网络的网络结构.
)pbdoc", pybind11::arg("attributes_or_structure"));

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
        .def("__call__", &initializers::RBFNormal::PyCall<matrix32, np_int32, float32>, R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"))
        .def("__call__", &initializers::RBFNormal::PyCall<matrix64, np_int64, float64>, R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"))
        .def("__call__", &initializers::RBFNormal::PyCall<matrix32, int32, float32>, R"pbdoc(
Arguments:
    hidden_units: int, 径向基函数网络的隐含层神经元数量.

Notes:
    - 这里隐含层神经元中心本应初始化为标准随机正态分布矩阵,
      但是实际工程发现, 有负值的时候可能会导致求高斯函数的时候增加损失不收敛,
      因此, 全部初始化为正数.
)pbdoc", pybind11::arg("hidden_units"));

    m.attr("__version__") = "backend.cc.initializers.0.6dev20211225";
}