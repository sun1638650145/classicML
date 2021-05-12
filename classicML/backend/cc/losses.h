//
// losses.h
// losses
//
// Create by 孙瑞琦 on 2020/1/25.
//
//

#ifndef LOSSES_H
#define LOSSES_H

#include <cmath>

#include "Eigen/Dense"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "exceptions.h"

namespace losses {
// 损失函数的基类.
class Loss {
    public:
        Loss();
        explicit Loss(std::string name);

        virtual double PyCall(const Eigen::MatrixXd &y_pred,
                              const Eigen::MatrixXd &y_true);

    public:
        std::string name;
};

// 二分类交叉熵损失函数.
class BinaryCrossentropy : public Loss {
    public:
        BinaryCrossentropy();
        explicit BinaryCrossentropy(std::string name);

        double PyCall(const Eigen::MatrixXd &y_pred,
                      const Eigen::MatrixXd &y_true) override;

    public:
        std::string name;
};

// 多分类交叉熵损失函数.
class CategoricalCrossentropy : public Loss {
    public:
        CategoricalCrossentropy();
        explicit CategoricalCrossentropy(std::string name);

        double PyCall(const Eigen::MatrixXd &y_pred,
                      const Eigen::MatrixXd &y_true) override;

    public:
        std::string name;
};

// 交叉熵损失函数.
class Crossentropy : public Loss {
    public:
        Crossentropy();
        explicit Crossentropy(std::string name);

        double PyCall(const Eigen::MatrixXd &y_pred,
                      const Eigen::MatrixXd &y_true) override;

    public:
        std::string name;
};

// 对数似然损失函数.
class LogLikelihood : public Loss {
    public:
        LogLikelihood();
        explicit LogLikelihood(std::string name);

        double PyCall(const Eigen::MatrixXd &y_true,
                      const Eigen::MatrixXd &beta,
                      const Eigen::MatrixXd &x_hat);

    public:
        std::string name;
};

// 均方误差损失函数.
class MeanSquaredError : public Loss {
    public:
        MeanSquaredError();
        explicit MeanSquaredError(std::string name);

        double PyCall(const Eigen::MatrixXd &y_pred,
                      const Eigen::MatrixXd &y_true) override;

    public:
        std::string name;
};
}  // namespace losses

PYBIND11_MODULE(losses, m) {
    m.doc() = R"pbdoc(classicML的损失函数, 以C++实现)pbdoc";

    // 注册自定义异常
    pybind11::register_exception<exceptions::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);

    pybind11::class_<losses::Loss>(m, "Loss", pybind11::dynamic_attr(), R"pbdoc(
损失函数的基类.

    Attributes:
        name: str, default=None,
            损失函数名称.

    Raises:
       NotImplementedError: __call__方法需要用户实现.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='loss',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default=None,
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &losses::Loss::name)
        .def("__call__", &losses::Loss::PyCall, pybind11::arg("y_pred"), pybind11::arg("y_true"));

    pybind11::class_<losses::BinaryCrossentropy, losses::Loss>(m, "BinaryCrossentropy", pybind11::dynamic_attr(), R"pbdoc(
二分类交叉熵损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='binary_crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='binary_crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &losses::BinaryCrossentropy::name)
        .def("__call__", &losses::BinaryCrossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<losses::CategoricalCrossentropy, losses::Loss>(m, "CategoricalCrossentropy", pybind11::dynamic_attr(), R"pbdoc(
多分类交叉熵损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='categorical_crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='categorical_crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &losses::CategoricalCrossentropy::name)
        .def("__call__", &losses::CategoricalCrossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<losses::Crossentropy, losses::Loss>(m, "Crossentropy", pybind11::dynamic_attr(), R"pbdoc(
交叉熵损失函数,
将根据标签的实际形状自动使用二分类或者多分类损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='crossentropy',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='crossentropy',
                损失函数名称.
)pbdoc", pybind11::arg("name"))
        .def_readonly("name", &losses::Crossentropy::name)
        .def("__call__", &losses::Crossentropy::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    pybind11::class_<losses::LogLikelihood, losses::Loss>(m, "LogLikelihood", pybind11::dynamic_attr(), R"pbdoc(
对数似然损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='log_likelihood',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='log_likelihood',
                损失函数名称.
)pbdoc")
        .def_readonly("name", &losses::LogLikelihood::name)
        .def("__call__", &losses::LogLikelihood::PyCall,
             pybind11::arg("y_true"),
             pybind11::arg("beta"),
             pybind11::arg("x_hat"));

    pybind11::class_<losses::MeanSquaredError, losses::Loss>(m, "MeanSquaredError", pybind11::dynamic_attr(), R"pbdoc(
均方误差损失函数.
)pbdoc")
        .def(pybind11::init(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def(pybind11::init<std::string>(), R"pbdoc(
        Arguments:
            name: str, default='mean_squared_error',
                损失函数名称.
)pbdoc")
        .def_readonly("name", &losses::MeanSquaredError::name)
        .def("__call__", &losses::MeanSquaredError::PyCall,
             pybind11::arg("y_pred"),
             pybind11::arg("y_true"));

    // Aliases.
    m.attr("MSE") = m.attr("MeanSquaredError");

    m.attr("__version__") = "backend.cc.losses.0.7.1";
}

#endif /* LOSSES_H */