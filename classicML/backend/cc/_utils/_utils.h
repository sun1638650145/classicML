//
// _utils.h
// _utils
//
// Created by 孙瑞琦 on 2021/6/26.
//
//

#ifndef CLASSICML_BACKEND_CC__UTILS__UTILS_H_
#define CLASSICML_BACKEND_CC__UTILS__UTILS_H_

#include <ctime>
#include <iostream>

#include "pybind11/pybind11.h"

#include "classicML/backend/cc/dtypes.h"

namespace _utils {
// 训练进度条.
class ProgressBar {
    public:
        ProgressBar(const uint32 &epochs, const pybind11::object &loss, const pybind11::object &metric);
        virtual ~ProgressBar() = default;

        void PyCall(const uint32 &epoch,
                    const float64 &current,
                    const float64 &loss_value,
                    const float64 &metric_value);

    private:
        void _update_info(const uint32 &epoch,
                          const float64 &current,
                          const float64 &loss_value,
                          const float64 &metric_value);
        void _dynamic_display();
        void _draw_bar(const uint32 &epoch);
        void _draw_detail(const uint32 &epoch,
                          const float64 &current,
                          const float64 &loss_value,
                          const float64 &metric_value);

    public:
        uint32 epochs;
        pybind11::object loss;
        pybind11::object metric;

    private:
        time_t ETD = time(nullptr);
        std::string info;
};
}  // namespace _utils

#endif /* CLASSICML_BACKEND_CC__UTILS__UTILS_H_ */
