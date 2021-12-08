//
// callbacks.h
// callbacks
//
// Created by 孙瑞琦 on 2021/6/11.
//
//

#ifndef CLASSICML_BACKEND_CC_CALLBACKS_H_
#define CLASSICML_BACKEND_CC_CALLBACKS_H_

#include "pybind11/stl.h"

#include "../dtypes.h"

namespace callbacks {
// 保存训练的历史记录.
class History {
    public:
        History();
        explicit History(std::string name, std::string loss_name, std::string metric_name);
        virtual ~History() = default;

        void PyCall(const float64 &loss_value, const float64 &metric_value);

    public:
        std::string name;
        std::string loss_name;
        std::string metric_name;
        std::vector<float64> loss;
        std::vector<float64> metric;
};
}  // namespace callbacks

#endif /* CLASSICML_BACKEND_CC_CALLBACKS_H_ */
