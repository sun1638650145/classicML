//
// activations.h
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#ifndef CLASSICML_BACKEND_CC_ACTIVATIONS_H_
#define CLASSICML_BACKEND_CC_ACTIVATIONS_H_

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "dtypes.h"
#include "exceptions.h"

namespace activations {
// 激活函数基类.
class Activation {
    public:
        Activation();
        explicit Activation(std::string name);
        virtual ~Activation() = default;

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);

    public:
        std::string name;
};

// ReLU激活函数.
class Relu : public Activation {
    public:
        Relu();
        explicit Relu(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};

// Sigmoid激活函数.
class Sigmoid : public Activation {
    public:
        Sigmoid();
        explicit Sigmoid(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};

// Softmax激活函数.
class Softmax : public Activation {
    public:
        Softmax();
        explicit Softmax(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};
}  // namespace activations

PYBIND11_MODULE(activations, m) {
    m.doc() = R"pbdoc(classicML的激活函数, 以C++实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<activations::Activation>(m, "Activation", R"pbdoc(
激活函数基类.

    Attributes:
        name: str, default='activation',
            激活函数名称.

    Raises:
       NotImplementedError: __call__, diff方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='activation',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='activation',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &activations::Activation::name)
        .def("__call__", &activations::Activation::PyCall<Eigen::MatrixXf>, pybind11::arg("z"))
        .def("__call__", &activations::Activation::PyCall<Eigen::MatrixXd>, pybind11::arg("z"))
        .def("diff", &activations::Activation::Diff<Eigen::MatrixXf>,
             pybind11::arg("output"), pybind11::arg("a"))
        .def("diff", &activations::Activation::Diff<Eigen::MatrixXd>,
             pybind11::arg("output"), pybind11::arg("a"));

    pybind11::class_<activations::Relu, activations::Activation>(m, "Relu", R"pbdoc(
ReLU激活函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='relu',
                激活函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='relu',
                激活函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &activations::Relu::name)
        .def("__call__", &activations::Relu::PyCall<Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("__call__", &activations::Relu::PyCall<Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &activations::Relu::Diff<Eigen::MatrixXf>, R"pbdoc(
ReLU函数的微分.

    Arguments:
        output: numpy.ndarray, 前向传播输出的张量.
        a: numpy.ndarray, 输入的张量.

    Notes:
        ReLU函数在大于零区间的导数应该是恒为一, 如果按此计算在实际应用上会随着训练轮数的增加, 最后模型的输出是一个随机概率,
        作者个人认为原因是随着轮数的增加, 大部分神经元都恒为激活态, 成为一个线性操作(缩放实际上相当于不参与计算了).
        在实际应用中使用原值发现可以避免这种想象.
)pbdoc", pybind11::arg("output"), pybind11::arg("a"))
        .def("diff", &activations::Relu::Diff<Eigen::MatrixXd>, R"pbdoc(
ReLU函数的微分.

    Arguments:
        output: numpy.ndarray, 前向传播输出的张量.
        a: numpy.ndarray, 输入的张量.

    Notes:
        ReLU函数在大于零区间的导数应该是恒为一, 如果按此计算在实际应用上会随着训练轮数的增加, 最后模型的输出是一个随机概率,
        作者个人认为原因是随着轮数的增加, 大部分神经元都恒为激活态, 成为一个线性操作(缩放实际上相当于不参与计算了).
        在实际应用中使用原值发现可以避免这种想象.
)pbdoc", pybind11::arg("output"), pybind11::arg("a"));

    pybind11::class_<activations::Sigmoid, activations::Activation>(m, "Sigmoid", R"pbdoc(
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
        .def_readwrite("name", &activations::Sigmoid::name)
        .def("__call__", &activations::Sigmoid::PyCall<Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("__call__", &activations::Sigmoid::PyCall<Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &activations::Sigmoid::Diff<Eigen::MatrixXf>, R"pbdoc(
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
)pbdoc",pybind11::arg("output"), pybind11::arg("a"))
        .def("diff", &activations::Sigmoid::Diff<Eigen::MatrixXd>, R"pbdoc(
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
)pbdoc",pybind11::arg("output"), pybind11::arg("a"));

    pybind11::class_<activations::Softmax, activations::Activation>(m, "Softmax", R"pbdoc(
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
        .def_readwrite("name", &activations::Softmax::name)
        .def("__call__", &activations::Softmax::PyCall<Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("__call__", &activations::Softmax::PyCall<Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        z: numpy.ndarray, 输入的张量.

    Returns:
        经过激活后的张量.
)pbdoc", pybind11::arg("z"))
        .def("diff", &activations::Softmax::Diff<Eigen::MatrixXf>, R"pbdoc(
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
)pbdoc",pybind11::arg("output"), pybind11::arg("a"))
        .def("diff", &activations::Softmax::Diff<Eigen::MatrixXd>, R"pbdoc(
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
    m.attr("relu") = activations::Relu();
    m.attr("sigmoid") = activations::Sigmoid();
    m.attr("softmax") = activations::Softmax();

    m.attr("__version__") = "backend.cc.activations.0.6.b0";
}

#endif /* CLASSICML_BACKEND_CC_ACTIVATIONS_H_ */