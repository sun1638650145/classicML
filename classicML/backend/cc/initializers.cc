//
// initializers.cc
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
//
//

#include "initializers.h"

initializers::Initializer::Initializer() {
    this->name = "initializer";
}

initializers::Initializer::Initializer(std::string name) {
    this->name = std::move(name);
}

initializers::Initializer::Initializer(std::string name, int seed) {
    this->name = std::move(name);
    this->seed = seed;
}

Eigen::MatrixXd initializers::Initializer::PyCall(const pybind11::args &args,
                                                  const pybind11::kwargs &kwargs) {
    throw exceptions::NotImplementedError();
}