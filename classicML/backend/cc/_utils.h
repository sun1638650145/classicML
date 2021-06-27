//
// _utils.h
// _utils
//
// Created by 孙瑞琦 on 2021/6/26.
//
//

#ifndef CLASSICML_BACKEND_CC__UTILS_H_
#define CLASSICML_BACKEND_CC__UTILS_H_

#include <ctime>
#include <iostream>

#include "pybind11/pybind11.h"

namespace _utils {
// 训练进度条.
class ProgressBar {
    public:
        ProgressBar(const unsigned int &epochs, const pybind11::object &loss, const pybind11::object &metric);
        virtual ~ProgressBar() = default;

        void PyCall(const unsigned int &epoch,
                    const double &current,
                    const double &loss_value,
                    const double &metric_value);

    private:
        void _update_info(const unsigned int &epoch,
                          const double &current,
                          const double &loss_value,
                          const double &metric_value);
        void _dynamic_display();
        void _draw_bar(const unsigned int &epoch);
        void _draw_detail(const unsigned int &epoch,
                          const double &current,
                          const double &loss_value,
                          const double &metric_value);

    public:
        unsigned int epochs;
        pybind11::object loss;
        pybind11::object metric;

    private:
        time_t ETD = time(nullptr);
        std::string info;
};
}  // namespace _utils

PYBIND11_MODULE(_utils, m) {
    m.doc() = R"pbdoc(classicML的工具类, 以C++实现)pbdoc";

    pybind11::class_<_utils::ProgressBar>(m, "ProgressBar", R"pbdoc(
训练进度条.

    Attributes:
        ETD: float, 优化器启动的时间戳.
        epochs: int, 训练的轮数.
        loss: str, classicML.losses.Loss 实例,
            模型使用的损失函数.
        metric: str, classicML.metrics.Metric 实例,
            模型使用的评估函数.
)pbdoc")
        .def(pybind11::init<unsigned int, pybind11::object, pybind11::object>(), R"pbdoc(
Arguments:
    epochs: int, 训练的轮数.
        loss: str, classicML.losses.Loss 实例,
            模型使用的损失函数.
        metric: str, classicML.metrics.Metric 实例,
            模型使用的评估函数.
)pbdoc",
             pybind11::arg("epochs"), pybind11::arg("loss"), pybind11::arg("metric"))
        .def_readonly("epochs", &_utils::ProgressBar::epochs)
        .def_readonly("loss", &_utils::ProgressBar::loss)
        .def_readonly("metric", &_utils::ProgressBar::metric)
        .def("__call__", &_utils::ProgressBar::PyCall, R"pbdoc(
函数实现.

    Arguments:
        epoch: int, 当前的训练轮数.
        current: float, 当前的时间戳.
        loss_value: float, 当前的损失值.
        metric_value: float, 当前的评估值.
)pbdoc",
             pybind11::arg("epoch"),
             pybind11::arg("current"),
             pybind11::arg("loss"),
             pybind11::arg("metric"));

    m.attr("__version__") = "backend.cc._utils.0.1.1";
}

#endif /* CLASSICML_BACKEND_CC__UTILS_H_ */
