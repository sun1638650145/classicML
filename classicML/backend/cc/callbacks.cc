//
// Created by 孙瑞琦 on 2021/6/11.
//

#include "callbacks.h"

callbacks::History::History() {
    this->name = "history";
    this->loss_name = "loss";
    this->metric_name = "metric";
}

callbacks::History::History(std::string name, std::string loss_name, std::string metric_name) {
    this->name = std::move(name);
    this->loss_name = std::move(loss_name);
    this->metric_name = std::move(metric_name);
}

// TODO(Steve R. Sun, tag:code): backend.cc.callbacks.0.1.2 的测试性能相较Python版本略慢0.3us.
// 记录训练过程中的历史记录, 输入为损失和评估函数值.
void callbacks::History::PyCall(const double &loss_value, const double &metric_value) {
    this->loss.push_back(loss_value);
    this->metric.push_back(metric_value);
}