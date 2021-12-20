//
// kernels.h
// kernels
//
// Create by 孙瑞琦 on 2021/2/10.
//
//

#ifndef CLASSICML_BACKEND_CC_KERNELS_H_
#define CLASSICML_BACKEND_CC_KERNELS_H_

#include "pybind11/eigen.h"

#include "../dtypes.h"
#include "../exceptions.h"
#include "../matrix_op.h"

namespace kernels {
// 损失函数的基类.
class Kernel {
    public:
        Kernel();
        explicit Kernel(std::string name);
        virtual ~Kernel() = default;

        template<typename Matrix> Matrix PyCall(const Matrix &x_i, const Matrix &x_j);

    public:
        std::string name;
};

// 线性核函数.
class Linear : public Kernel {
    public:
        Linear();
        explicit Linear(std::string name);

        template<typename Matrix> Matrix PyCall(const Matrix &x_i, const Matrix &x_j);
};

// 多项式核函数.
class Polynomial : public Kernel {
    public:
        Polynomial();
        explicit Polynomial(std::string name, float64 gamma, int32 degree);

        template<typename Matrix> Matrix PyCall(const Matrix &x_i, const Matrix &x_j);

    public:
        // TODO(Steve R. Sun, tag:trick): 首先说明, 在保证 64-bit 模式下精度的同时, 32-bit 模式下速度还是有10%的提高.
        //  在Python侧声明参数数据类型都是float, 但C++侧设置为float64可以使得精度得以保证; 你可能想问为什么构造函数没有使用float32,
        //  因为C++侧声明float32, 解释器会按float32接收, 自动转换float64类型时精度损失; int32和int64没有浮点数部分就没有这个问题,
        //  在满足性能的情况下选一个内存占用小的.
        float64 gamma;
        int64 degree;
};

// 径向基核函数.
class RBF : public Kernel {
    public:
        RBF();
        explicit RBF(std::string name, float64 gamma);

        template<typename Matrix> Matrix PyCall(const Matrix &x_i, const Matrix &x_j);

    public:
        float64 gamma;
};

// 高斯核函数.
class Gaussian : public RBF {
    public:
        Gaussian();
        explicit Gaussian(std::string name, float64 gamma);
};

// Sigmoid核函数.
class Sigmoid : public Kernel {
    public:
        Sigmoid();
        explicit Sigmoid(std::string name, float64 gamma, float64 beta, float64 theta);

        template<typename Matrix> Matrix PyCall(const Matrix &x_i, const Matrix &x_j);

    public:
        float64 gamma;
        float64 beta;
        float64 theta;
};
}  // namespace kernels

#endif /* CLASSICML_BACKEND_CC_KERNELS_H_ */