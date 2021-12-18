//
// metrics.h
// metrics
//
// Create by 孙瑞琦 on 2020/12/28.
//
//

#ifndef CLASSICML_BACKEND_CC_METRICS_METRICS_H_
#define CLASSICML_BACKEND_CC_METRICS_METRICS_H_

#include "pybind11/eigen.h"

#include "../dtypes.h"
#include "../exceptions.h"

namespace metrics {
// 评估函数的基类.
class Metric {
    public:
        Metric();
        explicit Metric(std::string name);
        virtual ~Metric() = default;

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);

    public:
        std::string name;
};

// 准确率评估函数.
class Accuracy : public Metric {
    public:
        Accuracy();
        explicit Accuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};

// 二分类准确率评估函数.
class BinaryAccuracy : public Metric {
    public:
        BinaryAccuracy();
        explicit BinaryAccuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};

// 多分类准确率评估函数.
class CategoricalAccuracy : public Metric {
    public:
        CategoricalAccuracy();
        explicit CategoricalAccuracy(std::string name);

        template<typename Dtype, typename Matrix> Dtype PyCall(const Matrix &y_pred, const Matrix &y_true);
};
}  // namespace metrics

#endif /* CLASSICML_BACKEND_CC_METRICS_METRICS_H_ */