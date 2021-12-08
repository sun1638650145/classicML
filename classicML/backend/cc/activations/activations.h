//
// activations.h
// activations
//
// Create by 孙瑞琦 on 2021/1/21.
//
//

#ifndef CLASSICML_BACKEND_CC_ACTIVATIONS_ACTIVATIONS_H_
#define CLASSICML_BACKEND_CC_ACTIVATIONS_ACTIVATIONS_H_

#include "pybind11/eigen.h"

#include "../dtypes.h"
#include "../exceptions.h"

namespace activations {
// 激活函数基类.
class Activation {
    public:
        Activation();
        explicit Activation(std::string name);
        virtual ~Activation() = default;

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);

    public:
        std::string name;
};

// ReLU激活函数.
class Relu : public Activation {
    public:
        Relu();
        explicit Relu(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};

// Sigmoid激活函数.
class Sigmoid : public Activation {
    public:
        Sigmoid();
        explicit Sigmoid(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};

// Softmax激活函数.
class Softmax : public Activation {
    public:
        Softmax();
        explicit Softmax(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &z);

        template<typename Matrix>
        Matrix Diff(const Matrix &output, const Matrix &a, const pybind11::args &args, const pybind11::kwargs &kwargs);
};
}  // namespace activations

#endif /* CLASSICML_BACKEND_CC_ACTIVATIONS_ACTIVATIONS_H_ */