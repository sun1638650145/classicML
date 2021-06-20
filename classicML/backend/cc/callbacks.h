//
// callbacks.h
// callbacks
//
// Created by 孙瑞琦 on 2021/6/11.
//
//

#ifndef CLASSICML_BACKEND_CC_CALLBACKS_H_
#define CLASSICML_BACKEND_CC_CALLBACKS_H_

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace callbacks {
// 保存训练的历史记录.
class History {
    public:
        History();
        explicit History(std::string name, std::string loss_name, std::string metric_name);
        virtual ~History() = default;

        void PyCall(const double &loss_value, const double &metric_value);

    public:
        std::string name;
        std::string loss_name;
        std::string metric_name;
        std::vector<double> loss;
        std::vector<double> metric;
};
}  // namespace callbacks

PYBIND11_MODULE(callbacks, m) {
    m.doc() = R"pbdoc(classicML的回调函数, 以C++实现)pbdoc";

    pybind11::class_<callbacks::History>(m, "History", R"pbdoc(
保存训练的历史记录.

    Attributes:
        name: str, default='history',
            历史记录的名称.
        loss_name: str, default='loss'
            使用损失函数的名称.
        metric_name: str, default='metric'
            使用评估函数的名称.
        loss: list, 损失值组成的列表
        metric: list, 评估值组成的列表

    Notes:
        - 使用历史记录数据, 会导致运行速度的降低.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
Arguments:
    name: str, default='history',
        历史记录的名称.
    loss_name: str, default='loss'
        使用损失函数的名称.
    metric_name: str, default='metric'
        使用评估函数的名称.
)pbdoc")
        .def(pybind11::init<std::string, std::string, std::string>(), R"pbdoc(
Arguments:
    name: str, default='history',
        历史记录的名称.
    loss_name: str, default='loss'
        使用损失函数的名称.
    metric_name: str, default='metric'
        使用评估函数的名称.
)pbdoc",
             pybind11::arg("name")="history",
             pybind11::arg("loss_name")="loss",
             pybind11::arg("metric_name")="metric")
        .def_readwrite("name", &callbacks::History::name)
        .def_readwrite("loss_name", &callbacks::History::loss_name)
        .def_readwrite("metric_name", &callbacks::History::metric_name)
        .def_readonly("loss", &callbacks::History::loss)
        .def_readonly("metric", &callbacks::History::metric)
        .def("__call__", &callbacks::History::PyCall, R"pbdoc(
记录当前的信息.
    Arguments:
        loss_value: float, 当前的损失值.
        metric_value: float, 当前的评估值.
)pbdoc",
            pybind11::arg("loss_value"), pybind11::arg("metric_name"));

    m.attr("__version__") = "backend.cc.callbacks.0.1.2";
}

#endif /* CLASSICML_BACKEND_CC_CALLBACKS_H_ */
