//
// metrics.h
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#ifndef CLASSICML_BACKEND_CC_METRICS_H_
#define CLASSICML_BACKEND_CC_METRICS_H_

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "dtypes.h"
#include "exceptions.h"

namespace metrics {
// 评估函数的基类.
class Metric {
    public:
        Metric();
        explicit Metric(std::string name);
        virtual ~Metric() = default;

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);

    public:
        std::string name;
};

// 准确率评估函数.
class Accuracy : public Metric {
    public:
        Accuracy();
        explicit Accuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};

// 二分类准确率评估函数.
class BinaryAccuracy : public Metric {
    public:
        BinaryAccuracy();
        explicit BinaryAccuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};

// 多分类准确率评估函数.
class CategoricalAccuracy : public Metric {
    public:
        CategoricalAccuracy();
        explicit CategoricalAccuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};
}  // namespace metrics

PYBIND11_MODULE(metrics, m) {
    m.doc() = R"pbdoc(classicML的评估函数, 以C++实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<metrics::Metric>(m, "Metric", R"pbdoc(
评估函数的基类.

    Attributes:
        name: str, default=None,
            评估函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
    Arguments:
            name: str, default='metric',
                评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
    Arguments:
            name: str, default='metric',
                评估函数名称.
)pbdoc",
             pybind11::arg("name"))
        .def_readwrite("name", &metrics::Metric::name)
        .def("__call__", &metrics::Metric::PyCall<float32, Eigen::MatrixXf>,
             pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::Metric::PyCall<float64, Eigen::MatrixXd>,
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::Accuracy, metrics::Metric>(m, "Accuracy", R"pbdoc(
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
        .def_readwrite("name", &metrics::Accuracy::name)
        .def("__call__", &metrics::Accuracy::PyCall<float32, Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::Accuracy::PyCall<float64, Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::BinaryAccuracy, metrics::Metric>(m, "BinaryAccuracy", R"pbdoc(
二分类准确率评估函数.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def_readwrite("name", &metrics::BinaryAccuracy::name)
        .def("__call__", &metrics::BinaryAccuracy::PyCall<float32, Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::BinaryAccuracy::PyCall<float64, Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::CategoricalAccuracy, metrics::Metric>(m, "CategoricalAccuracy", R"pbdoc(
多分类准确率评估函数.
)pbdoc")
        .def(pybind11::init())
        .def(pybind11::init<std::string>(), pybind11::arg("name"))
        .def_readwrite("name", &metrics::CategoricalAccuracy::name)
        .def("__call__", &metrics::CategoricalAccuracy::PyCall<float32, Eigen::MatrixXf>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::CategoricalAccuracy::PyCall<float64, Eigen::MatrixXd>, R"pbdoc(
    Arguments:
        y_pred: numpy.ndarray, 预测的标签.
        y_true: numpy.ndarray, 真实的标签.

    Returns:
        当前的准确率.
)pbdoc",
             pybind11::arg("y_pred"), pybind11::arg("y_true"));

    m.attr("__version__") = "backend.cc.metrics.0.4.b0";
}

#endif /* CLASSICML_BACKEND_CC_METRICS_H_ */