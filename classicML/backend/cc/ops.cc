//
//  ops.cc
//  ops
//
//  Created by 孙瑞琦 on 2020/10/10.
//
//

#include "ops.h"

// 返回KKT条件的违背值;
// 输入特征数据, 标签, 要计算的样本下标, 支持向量分类器使用的核函数, 全部拉格朗日乘子, 非零拉格朗日乘子和偏置项.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `Array` 兼容32位和64位浮点型Eigen::Array数组, `Vector` 兼容32位和64位浮点型向量,
// 不支持不同位数模板兼容.
// TODO(Steve R. Sun, tag:performance): 在CC中使用Python实现的核函数实际性能和Python没差别, 但是由于其他地方依旧使用的是C++代码, 还是会有明显的性能提高.
template<typename Matrix, typename Vector, typename Array>
Matrix ops::CalculateError(const Matrix &x,
                           const Matrix &y, // 列向量
                           const uint32 &i,
                           const pybind11::object &kernel,
                           const Matrix &alphas,  // 列向量
                           const Vector &non_zero_alphas,
                           const Matrix &b) {
    Matrix x_i = x.row(i);
    Matrix y_i = y.row(i);

    // 拉格朗日乘子全是零的时候, 每个样本都不会对结果产生影响.
    Matrix fx = b;
    // 如果有非零元素, 提取全部合格的标签和对应的拉格朗日乘子.
    if (non_zero_alphas.any()) {
        Matrix valid_x = matrix_op::GetNonZeroSubMatrix(x, non_zero_alphas);
        Array valid_y = matrix_op::GetNonZeroSubMatrix(y, non_zero_alphas);
        Matrix valid_alphas = matrix_op::GetNonZeroSubMatrix(alphas, non_zero_alphas);

        // 调用核函数的__call__方法.
        pybind11::object py_kappa = kernel(valid_x, x_i);
        auto kappa = py_kappa.cast<Matrix>();

        // 这里是Hadamard积, 故临时需要使用ArrayXd.
        Array temp = matrix_op::Reshape(valid_alphas, -1, 1);
        temp = temp * valid_y;
        auto matrix_temp = (Matrix)temp.transpose();

        fx = matrix_temp * kappa.transpose() + b;
    }

    Matrix error = fx - y_i;

    return error;
}

// 返回修剪后的拉格朗日乘子(32/64位), 输入拉格朗日乘子的下界和上界(float32/float64).
std::variant<Eigen::Array<float32, 1, 1>, Eigen::Array<float64, 1, 1>>
ops::ClipAlpha(const pybind11::buffer &alpha, const pybind11::buffer &low, const pybind11::buffer &high) {
    std::string type_code = alpha.request().format;
    if (type_code == "f") {
        auto _alpha = pybind11::cast<float32>(alpha);
        auto _low = pybind11::cast<float32>(low);
        auto _high = pybind11::cast<float32>(high);

        Eigen::Array<float32, 1, 1> clipped_alpha;
        clipped_alpha = _alpha;

        if (_alpha > _high) {
            clipped_alpha = _high;
        } else if (_alpha < _low) {
            clipped_alpha = _low;
        }

        return clipped_alpha;
    } else if (type_code == "d") {
        auto _alpha = pybind11::cast<float64>(alpha);
        auto _low = pybind11::cast<float64>(low);
        auto _high = pybind11::cast<float64>(high);

        Eigen::Array<float64, 1, 1> clipped_alpha;
        clipped_alpha = _alpha;

        if (_alpha > _high) {
            clipped_alpha = _high;
        } else if (_alpha < _low) {
            clipped_alpha = _low;
        }

        return clipped_alpha;
    }
    return {};
}

// 返回修剪后的拉格朗日乘子(32位), 输入拉格朗日乘子的下界和上界(Pure Python float).
Eigen::Array<float32, 1, 1> ops::ClipAlpha(const float32 &alpha, const float32 &low, const float32 &high) {
    Eigen::Array<float32, 1, 1> clipped_alpha;
    clipped_alpha = alpha;

    if (alpha > high) {
        clipped_alpha = high;
    } else if (alpha < low) {
        clipped_alpha = low;
    }

    return clipped_alpha;
}

// 获取类条件概率, 输入某个属性值的样本总数, 某个类别的样本总数, 类别的数量和是否使用平滑.
float64 ops::GetConditionalProbability(const uint32 &samples_on_attribute,
                                       const uint32 &samples_in_category,
                                       const uint32 &num_of_categories,
                                       const bool &smoothing) {
    if (smoothing) {
        return ((float64)samples_on_attribute + 1.0) / (float64)(samples_in_category + num_of_categories);
    } else {
        return (float64)samples_on_attribute / (float64)samples_in_category;
    }
}

// 获取有依赖的类先验概率, 输入类别为c的属性i上取值为xi的样本, 样本的总数, 特征数据和是否使用平滑.
float64 ops::GetDependentPriorProbability(const uint32 &samples_on_attribute_in_category,
                                          const uint32 &number_of_sample,
                                          const uint32 &values_on_attribute,
                                          const bool &smoothing) {
    float64 probability;

    if (smoothing) {
        probability = (float64)(samples_on_attribute_in_category + 1)
                / (float64)(number_of_sample + 2 * values_on_attribute);
    } else {
        probability = (float64)samples_on_attribute_in_category / (float64)number_of_sample;
    }

    return probability;
}

// 获取类先验概率, 输入特征数据, 标签和是否使用平滑.
// `Dtype` 兼容32位和64位整型, `RowVector` 兼容32位和64位整型行向量, 不支持不同位数模板兼容.
template<typename Dtype, typename RowVector>
std::tuple<Dtype, Dtype> ops::GetPriorProbability(const uint32 &number_of_sample,
                                                  const RowVector &y,
                                                  const bool &smoothing) {
    // 遍历获得反例的个数.
    uint32 num_of_negative_sample = y.size() - matrix_op::NonZero(y).size();

    Dtype p_0;
    if (smoothing) {
        p_0 = (num_of_negative_sample + 1.0) / (number_of_sample + 2);
    } else {
        p_0 = (Dtype)num_of_negative_sample / number_of_sample;
    }

    std::tuple<Dtype, Dtype> probability(p_0, 1 - p_0);

    return probability;
}


// 获取概率密度(32/64位), 输入样本的取值(float32/float64),
// 样本在某个属性的上的均值(float32/float64)和样本在某个属性上的方差(float32/float64).
std::variant<float32, float64> ops::GetProbabilityDensity(const pybind11::buffer &sample,
                                                          const pybind11::buffer &mean,
                                                          const pybind11::buffer &var) {
    std::string type_code = sample.request().format;
    if (type_code == "f") {
        auto _sample = pybind11::cast<float32>(sample);
        auto _mean = pybind11::cast<float32>(mean);
        auto _var = pybind11::cast<float32>(var);

        float32 probability =  1 / (sqrtf(2 * M_PI) * _var) * expf(-(powf((_sample - _mean), 2) / (2 * powf(_var, 2))));

        // probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.
        if (probability == 0) {
            probability = 1e-36;
        }

        return probability;
    } else if (type_code == "d") {
        auto _sample = pybind11::cast<float64>(sample);
        auto _mean = pybind11::cast<float64>(mean);
        auto _var = pybind11::cast<float64>(var);

        float64 probability =  1 / (sqrt(2 * M_PI) * _var) * exp(-(pow((_sample - _mean), 2) / (2 * pow(_var, 2))));

        // probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.
        if (probability == 0) {
            probability = 1e-36;
        }

        return probability;
    }

    return {};
}

// 获取概率密度(32位), 输入样本的取值(Pure Python float),
// 样本在某个属性的上的均值(Pure Python float)和样本在某个属性上的方差(Pure Python float).
float32 ops::GetProbabilityDensity(const float32 &sample, const float32 &mean, const float32 &var) {
    float32 probability =  1 / (sqrtf(2 * M_PI) * var) * expf(-(powf((sample - mean), 2) / (2 * powf(var, 2))));

    // probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.
    if (probability == 0) {
        probability = 1e-36;
    }

    return probability;
}

// 返回投影向量, 输入为类内散度矩阵和反正例的均值向量.
Eigen::MatrixXd ops::GetW(const Eigen::MatrixXd &S_w, const Eigen::MatrixXd &mu_0, const Eigen::MatrixXd &mu_1) {
    pybind11::print("WARNING:classicML:`ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.");
    // 公式(数学公式难以表示, 使用latex语法): w = (S_w)^{-1}(\mu_0 - \mu_1)
    Eigen::MatrixXd S_w_inv = S_w.inverse();
    Eigen::MatrixXd mu_t = (mu_0 - mu_1).transpose();

    Eigen::MatrixXd w = S_w_inv * mu_t;

    return matrix_op::Reshape(w, 1, -1);
}

// 返回投影向量, 输入为类内散度矩阵和反正例的均值向量.
// `Matrix` 兼容的32位和64位浮点型Eigen::Matrix矩阵.
template <typename Matrix>
Matrix ops::GetW_V2(const Matrix &S_w, const Matrix &mu_0, const Matrix &mu_1) {
    // 分解奇异值.
    Eigen::BDCSVD<Matrix> svd(S_w, Eigen::ComputeFullU|Eigen::ComputeFullV);
    const Matrix &U = svd.matrixU();
    const Matrix &Sigma = svd.singularValues();
    const Matrix &V_t = svd.matrixV();
    Matrix Sigma_mat = Sigma.asDiagonal();

    // 公式(使用latex语法): w = (S_w)^{-1}(\mu_0 - \mu_1)
    Matrix S_w_inv = V_t.transpose() * Sigma_mat * U.transpose();
    Matrix mu_t = (mu_0 - mu_1).transpose();

    Matrix w = S_w_inv * mu_t;

    return matrix_op::Reshape(w, 1, -1);
}

// 返回类内散度矩阵, 输入为反正例的集合矩阵和反正例的均值向量.
// `Matrix` 兼容的32位和64位浮点型Eigen::Matrix矩阵.
template <typename Matrix>
Matrix ops::GetWithinClassScatterMatrix(const Matrix &X_0,
                                        const Matrix &X_1,
                                        const Matrix &mu_0,
                                        const Matrix &mu_1) {
    // 公式(使用latex语法):
    // S_w = \sum_0 + \sum_1
    // \sum_i = \sum_{x \in X_i}(x - \mu_0)(x - \mu_1)^T
    Matrix S_0 = (matrix_op::BroadcastSub(X_0, mu_0)).transpose() * (matrix_op::BroadcastSub(X_0, mu_0));
    Matrix S_1 = (matrix_op::BroadcastSub(X_1, mu_1)).transpose() * (matrix_op::BroadcastSub(X_1, mu_1));

    Matrix S_w = S_0 + S_1;

    return S_w;
}

// 返回第二个拉格朗日乘子的下标和违背值组成的元组, 输入KKT条件的违背值, KKT条件的违背值缓存和非边界拉格朗日乘子.
// `Dtype` 兼容32位和64位浮点数, `RowVector` 兼容32位和64位浮点型行向量,
// 不支持不同位数模板兼容.
// Python的内置类型只有`float`, 这里和使用`pybind11::buffer`最大的区别是在于, Python侧`float`和`double`无法区分, 但是此处重载时
// 会根据行向量精准匹配, C++侧会按照`float`计算, `float`自动类型转换`double`使得Python侧`pure float`返回值恰巧结果为8-15位随机值.
// [内置类型](https://docs.python.org/zh-cn/3.8/library/stdtypes.html)
template <typename Dtype, typename RowVector>
std::tuple<uint32, Dtype> ops::SelectSecondAlpha(const Dtype &error,
                                                 const RowVector &error_cache,
                                                 const RowVector &non_bound_alphas) {
    std::vector<uint32> non_bound_index = matrix_op::NonZero(non_bound_alphas);

    uint32 index_alpha = 0;
    Dtype error_alpha = error_cache[index_alpha];
    Dtype delta_e = abs(error - error_cache[non_bound_index[0]]);

    // 选取最大间隔的拉格朗日乘子对应的下标和违背值.
    for (uint32 i : non_bound_index) {
        Dtype temp = abs(error - error_cache[i]);
        if (temp > delta_e) {
            delta_e = temp;
            index_alpha = i;
            error_alpha = error_cache[index_alpha];
        }
    }

    std::tuple<uint32, Dtype> alpha_tuple(index_alpha, error_alpha);

    return alpha_tuple;
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 只处理float64的输入数据.
std::string ops::TypeOfTarget(const Eigen::MatrixXd &y) {
    pybind11::print("WARNING:classicML: `ops.cc_type_of_target` 已经被弃用,"
                    " 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.");
    bool any = true;
    std::set<float64> buffer;

    // 任何一个元素取整不等于它本身的就是连续值.
    for (int32 row = 0; row < y.rows(); row ++) {
        for (int32 col = 0; col < y.cols(); col ++) {
            buffer.insert(y(row, col));
            if (y(row, col) != (int32)y(row, col)) {
                any = false;
            }
        }
    }
    if (!any) {
        return "continuous";
    }

    // 取唯一值统计元素个数.
    if (y.cols() == 1) {
        if (buffer.size() == 2) {
            return "binary";
        } else if (buffer.size() > 2) {
            return "multiclass";
        }
    }

    // 行数不为一, 且元素个数超过二.
    if (buffer.size() >= 2) {
        return "multilabel";
    }

    return "unknown";
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 只处理int64的输入数据.
std::string ops::TypeOfTarget(const Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic> &y) {
    pybind11::print("WARNING:classicML: `ops.cc_type_of_target` 已经被弃用,"
                    " 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.");
    // 取唯一值统计元素个数.
    std::set<int64> buffer;
    for (int32 row = 0; row < y.rows(); row ++) {
        for (int32 col = 0; col < y.cols(); col ++) {
            buffer.insert(y(row, col));
        }
    }
    if (y.cols() == 1) {
        if (buffer.size() == 2) {
            return "binary";
        } else if (buffer.size() > 2) {
            return "multiclass";
        }
    }

    // 行数不为一, 且元素个数超过二.
    if (buffer.size() >= 2) {
        return "multilabel";
    }

    return "unknown";
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 处理其他类型的输入数据.
// TODO(Steve R. Sun, tag:code): Python版本的和CC版本在对于判断str类型的有差异, CC版本全部返回的是unknown.
std::string ops::TypeOfTarget(const pybind11::array &y) {
    pybind11::print("WARNING:classicML: `ops.cc_type_of_target` 已经被弃用,"
                    " 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.");
    return "unknown";
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 相比于`ops::TypeOfTarget`可以处理单字符的输入数据.
std::string ops::TypeOfTarget_V2(const pybind11::array &y) {
    if ((std::string)pybind11::str(y.dtype()) == "object" or y.ndim() > 2) {
        return "unknown";
    }

    if (y.dtype().kind() == 'f' and !matrix_op::AnyDiscreteInteger(y)) {
        return "continuous";
    }

    // `matrix_op::Unique`返回是一个`std::variant`, 需要使用`std::get`处理.
    int32 num_unique;
    if (y.dtype().kind() == 'f' or y.dtype().kind() == 'i') {
        num_unique = (int32)std::get<std::set<float32>>(matrix_op::Unique(y)).size();
    } else if (y.dtype().kind() == 'U') {
        num_unique = (int32)std::get<std::set<uint8>>(matrix_op::Unique(y)).size();
    }

    if (y.ndim() == 1 or y.request().shape[1] == 1) {
        if (num_unique == 2) {
            return "binary";
        } else if (num_unique > 2) {
            return "multiclass";
        }
    }

    if (y.ndim() == 2 and num_unique >= 2) {
        return "multilabel";
    }

    return "unknown";
}