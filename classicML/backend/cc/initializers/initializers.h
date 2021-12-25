//
// initializers.h
// initializers
//
// Create by 孙瑞琦 on 2021/5/12.
// Refactor by 孙瑞琦 on 2021/12/25.
//
//

#ifndef CLASSICML_BACKEND_CC_INITIALIZERS_INITIALIZERS_H_
#define CLASSICML_BACKEND_CC_INITIALIZERS_INITIALIZERS_H_

#include "pybind11/stl.h"

#include "../dtypes.h"
#include "../exceptions.h"
#include "../matrix_op.h"
#include "../numpy_patch.h"

namespace initializers {
// 初始化器的基类.
class Initializer {
    public:
        Initializer();
        explicit Initializer(std::string name);
        explicit Initializer(std::string name, std::optional<uint32> seed);
        virtual ~Initializer() = default;

        template<typename Matrix>
        Matrix PyCall(const pybind11::args &args, const pybind11::kwargs &kwargs);

    public:
        std::string name;
        std::optional<uint32> seed;
};

// 正态分布随机初始化器.
class RandomNormal: public Initializer {
    public:
        RandomNormal();
        explicit RandomNormal(std::string name);
        explicit RandomNormal(std::string name, std::optional<uint32> seed);

        // TODO(Steve R. Sun, tag:code): 这里不是真正上的C++ overload, 由于模板参数数量相同, 因此只在python侧使用了相同的函数名.
        // overload
        template<typename Matrix, typename Int, typename Float>
        Matrix PyCall1(Int &attributes_or_structure);

        template<typename Matrix, typename RowVector, typename Float>
        std::map<std::string, Matrix> PyCall2(const RowVector &attributes_or_structure);
};

// He正态分布随机初始化器.
class HeNormal: public Initializer {
    public:
        HeNormal();
        explicit HeNormal(std::string name);
        explicit HeNormal(std::string name, std::optional<uint32> seed);

        // overload
        template<typename Matrix, typename Int, typename Float>
        Matrix PyCall1(Int &attributes_or_structure);

        template<typename Matrix, typename RowVector, typename Float>
        std::map<std::string, Matrix> PyCall2(const RowVector &attributes_or_structure);
};

// Xavier正态分布随机初始化器.
class XavierNormal: public Initializer {
    public:
        XavierNormal();
        explicit XavierNormal(std::string name);
        explicit XavierNormal(std::string name, std::optional<uint32> seed);

        // overload
        template<typename Matrix, typename Int, typename Float>
        Matrix PyCall1(Int &attributes_or_structure);

        template<typename Matrix, typename RowVector, typename Float>
        std::map<std::string, Matrix> PyCall2(const RowVector &attributes_or_structure);
};

// Glorot正态分布随机初始化器.
class GlorotNormal: public XavierNormal {
    public:
        GlorotNormal();
        explicit GlorotNormal(std::string name);
        explicit GlorotNormal(std::string name, std::optional<uint32> seed);
};

// RBF网络的初始化器.
class RBFNormal: public Initializer {
    public:
        RBFNormal();
        explicit RBFNormal(std::string name);
        explicit RBFNormal(std::string name, std::optional<uint32> seed);

        // overload
        template<typename Matrix, typename Int, typename Float>
        std::map<std::string, Matrix> PyCall(Int &hidden_units);
};
} // namespace initializers

#endif /* CLASSICML_BACKEND_CC_INITIALIZERS_INITIALIZERS_H_ */