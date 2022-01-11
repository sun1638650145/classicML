//
// metrics_wrapper.cc
// metrics
//
// Created by 孙瑞琦 on 2021/12/18.
//
//

#include "pybind11/pybind11.h"

#include "metrics.h"

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
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &metrics::Metric::name)
        .def("__call__", &metrics::Metric::PyCall<float32, matrix32>, pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::Metric::PyCall<float64, matrix64>, pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::Accuracy, metrics::Metric>(m, "Accuracy", R"pbdoc(
准确率评估函数,
将根据标签的实际形状自动使用二分类或者多分类评估函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
Arguments:
    name: str, default='accuracy',
        评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
Arguments:
    name: str, default='accuracy',
        评估函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &metrics::Accuracy::name)
        .def("__call__", &metrics::Accuracy::PyCall<float32, matrix32>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::Accuracy::PyCall<float64, matrix64>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::BinaryAccuracy, metrics::Metric>(m, "BinaryAccuracy", R"pbdoc(
二分类准确率评估函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
Arguments:
    name: str, default='binary_accuracy',
        评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
Arguments:
    name: str, default='binary_accuracy',
        评估函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &metrics::BinaryAccuracy::name)
        .def("__call__", &metrics::BinaryAccuracy::PyCall<float32, matrix32>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::BinaryAccuracy::PyCall<float64, matrix64>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<metrics::CategoricalAccuracy, metrics::Metric>(m, "CategoricalAccuracy", R"pbdoc(
多分类准确率评估函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
Arguments:
    name: str, default='categorical_accuracy',
        评估函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
Arguments:
    name: str, default='categorical_accuracy',
        评估函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readwrite("name", &metrics::CategoricalAccuracy::name)
        .def("__call__", &metrics::CategoricalAccuracy::PyCall<float32, matrix32>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"))
        .def("__call__", &metrics::CategoricalAccuracy::PyCall<float64, matrix64>, R"pbdoc(
Arguments:
    y_pred: numpy.ndarray, 预测的标签.
    y_true: numpy.ndarray, 真实的标签.

Returns:
    当前的准确率.
)pbdoc", pybind11::arg("y_pred"), pybind11::arg("y_true"));

    m.attr("__version__") = "backend.cc.metrics.0.4.2";
}