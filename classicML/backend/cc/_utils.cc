//
// _utils.cc
// _utils
//
// Created by 孙瑞琦 on 2021/6/26.
//
//

#include "_utils.h"

_utils::ProgressBar::ProgressBar(const unsigned int &epochs,
                                 const pybind11::object &loss,
                                 const pybind11::object &metric) {
    this->epochs = epochs;
    this->loss = loss;
    this->metric = metric;
}

// 输入当前的训练轮数, 当前的时间戳, 当前的损失值和当前的评估值.
void _utils::ProgressBar::PyCall(const unsigned int &epoch,
                                 const double &current,
                                 const double &loss_value,
                                 const double &metric_value) {
    this->_update_info(epoch, current, loss_value, metric_value);
    this->_dynamic_display();
}

// 在终端上显示进度条, 输入当前的训练轮数, 当前的时间戳, 当前的损失值和当前的评估值.
void _utils::ProgressBar::_update_info(const unsigned int &epoch,
                                       const double &current,
                                       const double &loss_value,
                                       const double &metric_value) {
    this->_draw_bar(epoch);
    this->_draw_detail(epoch, current, loss_value, metric_value);
}

// 在终端显示.
void _utils::ProgressBar::_dynamic_display() {
    std::cout << "\r" << this->info;
}

// 绘制进度条(无论总的训练轮数是多少, 显示条的总更新步数是25次), 输入当前的训练轮数.
void _utils::ProgressBar::_draw_bar(const unsigned int &epoch) {
    this->info = "Epoch " + std::to_string(epoch) + "/" + std::to_string(this->epochs) + " [";

    if (epoch > (this->epochs - (this->epochs / 25))) {
        this->info += "=========================]";
    } else {
        // 获取箭头的实时位置.
        // arrow = epoch / (epochs / 25)
        int arrow = ceil((double)epoch / this->epochs * 25);

        for (int i = 0; i < arrow - 1; i ++) {
            this->info += "=";
        }
        this->info += ">";
        for (int i = 0; i < 25 - arrow; i ++) {
            this->info += ".";
        }
        this->info += "]";
    }
}

// 绘制显示的计算信息, 输入当前的训练轮数, 当前的时间戳, 当前的损失值和当前的评估值.
void _utils::ProgressBar::_draw_detail(const unsigned int &epoch,
                                       const double &current,
                                       const double &loss_value,
                                       const double &metric_value) {
    // 时间信息.
    if (epoch == 0) {
        this->info += " ETA: 00:00";
    } else if (epoch > (this->epochs - (this->epochs / 25))) {
        long total = time(nullptr) - this->ETD;
        long per_epoch = total * 1000 / this->epochs;
        this->info += (" " + std::to_string(total) + "s " + std::to_string(per_epoch) + "ms/step");
    } else {
        long ETA = (time(nullptr) - (long)current) * (this->epochs - epoch);  // 剩余时间.
        if (ETA < 60) {
            this->info += ("ETA: " + std::to_string(ETA) + "s");
        } else {
            long ETA_minutes = floor((double)ETA / 60);
            long ETA_seconds = ETA % 60;
            this->info += ("ETA: " + std::to_string(ETA_minutes) + ":" + std::to_string(ETA_seconds));
        }
    }

    // 数值信息.
    std::string loss_name = getattr(this->loss, "name").cast<std::string>();
    std::string _loss_value = std::to_string(loss_value);
    _loss_value = _loss_value.substr(0, _loss_value.find('.') + 4 + 1);

    std::string metric_name = getattr(this->metric, "name").cast<std::string>();
    std::string _metric_value = std::to_string(metric_value);
    _metric_value = _metric_value.substr(0, _metric_value.find('.') + 4 + 1);

    this->info += (" - " + loss_name + ": " + _loss_value + " - " + metric_name + ": " + _metric_value);

    if (epoch == this->epochs) {
        this->info += '\n';
    }
}
