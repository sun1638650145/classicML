//
// ops.cc
// ops
//
// Created by 孙瑞琦 on 2020/10/10.
// Refactor by 孙瑞琦 on 2021/12/29.
//
//

#include "ops.h"

namespace ops {
// 返回自助采样后的新样本.
// 输入数据样本, (对应标签)和(随机种子).
// `XMatrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
// `YMatrix` 兼容32位和64位浮点型, 32位和64位整型Eigen::Matrix矩阵.
template<typename XMatrix, typename YMatrix>
std::tuple<XMatrix, YMatrix> BootstrapSampling1(const XMatrix &x, const YMatrix &y, std::optional<uint32> seed) {
    auto num_of_samples = (int32)x.rows();

    // 检查样本的第一维是否一致.
    if (num_of_samples != y.rows()) {
        std::string error_message = "两个数组长度不一致[" + std::to_string(num_of_samples)
                                    + ", " + std::to_string(y.rows()) + "].";
        throw pybind11::value_error(error_message);
    }

    // 进行随机采样, 生成索引.
    srand(!seed.has_value() ? time(nullptr) : seed.value()); // 设置随机种子.
    row_vector32i indices = row_vector32i::Random(num_of_samples).unaryExpr(
        [=](int32 element) {
            return abs(element) % num_of_samples;
        }
    );

    // 进行切片.
    return {x(indices, Eigen::all), y(indices, Eigen::all)};
}

// 返回自助采样后的新样本.
// 输入数据样本, (对应标签)和(随机种子).
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix BootstrapSampling2(const Matrix &x, const pybind11::object &y, std::optional<uint32> seed) {
    auto num_of_samples = (int32)x.rows();

    // 进行随机采样, 生成索引.
    srand(!seed.has_value() ? time(nullptr) : seed.value()); // 设置随机种子.
    row_vector32i indices = row_vector32i::Random(num_of_samples).unaryExpr(
        [=](int32 element) {
            return abs(element) % num_of_samples;
        }
    );

    // 进行切片.
    return x(indices, Eigen::all);
}

// 返回KKT条件的违背值;
// 输入特征数据, 标签, 要计算的样本下标, 支持向量分类器使用的核函数, 全部拉格朗日乘子, 非零拉格朗日乘子和偏置项.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵, `Vector` 兼容32位和64位浮点型向量, `Array` 兼容32位和64位浮点型Eigen::Array数组,
// 不支持不同位数模板兼容.
// TODO(Steve R. Sun, tag:performance): 在C++中使用Python实现的核函数实际性能和Python没差别,
//  但是由于其他地方依旧使用的是C++代码, 还是会有明显的性能提高.
template<typename Matrix, typename Vector, typename Array>
Matrix CalculateError(const Matrix &x,
                      const Vector &y,
                      const uint32 &i,
                      const pybind11::object &kernel,
                      const Vector &alphas,
                      const Vector &non_zero_alphas,
                      const Matrix &b) {
    Matrix x_i = x.row(i);
    Vector y_i = y.row(i);

    // 拉格朗日乘子全是零的时候, 每个样本都不会对结果产生影响.
    Matrix fx = b;
    // 如果有非零元素, 提取全部合格的标签和对应的拉格朗日乘子.
    if (non_zero_alphas.any()) {
        auto valid_x = matrix_op::GetNonZeroSubMatrix<Matrix, Matrix, Vector>(x, non_zero_alphas);
        Array valid_y = matrix_op::GetNonZeroSubMatrix<Matrix, Vector, Vector>(y, non_zero_alphas);
        auto valid_alphas = matrix_op::GetNonZeroSubMatrix<Matrix, Vector, Vector>(alphas, non_zero_alphas);

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

// 返回欧式距离, 输入要计算欧式距离的两个值.
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix CalculateEuclideanDistance(const Matrix &x0, const Matrix &x1) {
    auto distances = Matrix(x0.rows(), x1.rows());

    // Eigen::Matrix 不能增加到3个维度, 只能使用循环实现.
    for (int row = 0; row < distances.rows(); row ++) {
        for (int col = 0; col < distances.cols(); col ++) {
            distances(row, col) = (x0.row(row) - x1.row(col)).norm();
        }
    }

    return distances;
}

// 返回修剪后的拉格朗日乘子(32/64位), 输入拉格朗日乘子的下界和上界(pure float/np.float32/np.float64).
template<typename RFloat, typename PFloat>
RFloat ClipAlpha(PFloat &alpha, PFloat &low, PFloat &high) {
    RFloat clipped_alpha = alpha;

    if (alpha > high) {
        clipped_alpha = high;
    } else if (alpha < low) {
        clipped_alpha = low;
    }

    return clipped_alpha;
}

// 返回差异向量, 输入比较差异的两个值(32/64位浮点型Eigen::Matrix矩阵.)和最小差异阈值(pure float/np.float32/np.float64).
// 不支持不同位数模板兼容.
template<typename Matrix, typename Float>
row_vector_bool CompareDifferences(const Matrix &x0, const Matrix &x1, Float &tol) {
    auto differences = (x0 - x1).array().abs();

    return differences.rowwise().maxCoeff() > tol; // 最大值大于最小阈值即可(即np.any()).
}

// 返回簇标记, 输入距离矩阵;
// `Matrix` 兼容32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
row_vector32i GetCluster(const Matrix &distances) {
    return matrix_op::ArgMin(distances);
}

// 获取类条件概率(32/64位), 输入某个属性值的样本总数, 某个类别的样本总数, 类别的数量和是否使用平滑(pure uint/np.uint32/np.uint64).
template<typename Float, typename Uint>
Float GetConditionalProbability(Uint &samples_on_attribute,
                                Uint &samples_in_category,
                                Uint &num_of_categories,
                                const bool &smoothing) {
    if (smoothing) {
        return ((Float)samples_on_attribute + 1.0) / (Float)(samples_in_category + num_of_categories);
    } else {
        return (Float)samples_on_attribute / (Float)samples_in_category;
    }
}

// 获取有依赖的类先验概率(32/64位),
// 输入类别为c的属性i上取值为xi的样本, 样本的总数, 特征数据和是否使用平滑(pure uint/np.uint32/np.uint64).
template<typename Float, typename Uint>
Float GetDependentPriorProbability(Uint &samples_on_attribute_in_category,
                                   Uint &number_of_sample,
                                   Uint &values_on_attribute,
                                   const bool &smoothing) {
    if (smoothing) {
        return (Float)(samples_on_attribute_in_category + 1) / (Float)(number_of_sample + 2 * values_on_attribute);
    } else {
        return (Float)samples_on_attribute_in_category / (Float)number_of_sample;
    }
}

// 获取类先验概率, 输入特征数据, 标签和是否使用平滑.
// `Dtype` 兼容32位和64位整型, `RowVector` 兼容32位和64位整型行向量, 不支持不同位数模板兼容.
template<typename Dtype, typename RowVector>
std::tuple<Dtype, Dtype> GetPriorProbability(const uint32 &number_of_sample,
                                             const RowVector &y,
                                             const bool &smoothing) {
    // 遍历获得反例的个数.
    uint32 num_of_negative_sample = y.size() - matrix_op::NonZero(y).size();

    Dtype p_0{};
    if (smoothing) {
        p_0 = (num_of_negative_sample + 1.0) / (number_of_sample + 2);
    } else {
        p_0 = (Dtype)num_of_negative_sample / number_of_sample;
    }

    std::tuple<Dtype, Dtype> probability(p_0, 1 - p_0);

    return probability;
}

// 获取概率密度(32/64位),
// 输入样本的取值(pure float/np.float32/np.float64),
// 样本在某个属性的上的均值(pure float/np.float32/np.float64)和样本在某个属性上的方差(pure float/np.float32/np.float64).
template<typename RFloat, typename PFloat>
RFloat GetProbabilityDensity(PFloat &sample, PFloat &mean, PFloat &var) {
    RFloat probability{};

    if (sizeof(sample) == sizeof(np_float32) or sizeof(sample) == sizeof(float32)) {
        probability = 1 / (sqrtf(2 * M_PI) * var) * expf(-(powf((sample - mean), 2) / (2 * powf(var, 2))));
    } else {
        probability = 1 / (sqrt(2 * M_PI) * var) * exp(-(pow((sample - mean), 2) / (2 * pow(var, 2))));
    }

    // probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.
    if (probability == 0) {
        probability = 1e-36;
    }

    return probability;
}

// 返回投影向量, 输入为类内散度矩阵和反正例的均值向量.
matrix64 GetW(const matrix64 &S_w, const matrix64 &mu_0, const matrix64 &mu_1) {
    pybind11::print("WARNING:classicML:`ops.cc_get_w` 已经被弃用, 它将在未来的正式版本中被移除, 请使用 `ops.cc_get_w_v2`.");
    // 公式(数学公式难以表示, 使用latex语法): w = (S_w)^{-1}(\mu_0 - \mu_1)
    matrix64 S_w_inv = S_w.inverse();
    matrix64 mu_t = (mu_0 - mu_1).transpose();

    matrix64 w = S_w_inv * mu_t;

    return matrix_op::Reshape(w, 1, -1);
}

// 返回投影向量, 输入为类内散度矩阵和反正例的均值向量.
// `Matrix` 兼容的32位和64位浮点型Eigen::Matrix矩阵.
template<typename Matrix>
Matrix GetW_V2(const Matrix &S_w, const Matrix &mu_0, const Matrix &mu_1) {
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
template<typename Matrix>
Matrix GetWithinClassScatterMatrix(const Matrix &X_0, const Matrix &X_1, const Matrix &mu_0, const Matrix &mu_1) {
    // 公式(使用latex语法):
    // S_w = \sum_0 + \sum_1
    // \sum_i = \sum_{x \in X_i}(x - \mu_0)(x - \mu_1)^T
    Matrix S_0 = (matrix_op::BroadcastSub(X_0, mu_0)).transpose() * (matrix_op::BroadcastSub(X_0, mu_0));
    Matrix S_1 = (matrix_op::BroadcastSub(X_1, mu_1)).transpose() * (matrix_op::BroadcastSub(X_1, mu_1));

    Matrix S_w = S_0 + S_1;

    return S_w;
}

// 返回第二个拉格朗日乘子的下标和违背值组成的元组, 输入KKT条件的违背值, KKT条件的违背值缓存和非边界拉格朗日乘子.
// `Dtype` 兼容(pure float/np.float32/np.float64, `RowVector` 兼容32位和64位浮点型行向量,
// 不支持不同位数模板兼容.
template<typename Dtype, typename RowVector>
std::tuple<uint32, Dtype> SelectSecondAlpha(Dtype &error,
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

    return {index_alpha, error_alpha};
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 只处理float64的输入数据.
std::string TypeOfTarget(const matrix64 &y) {
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
std::string TypeOfTarget(const matrix64i &y) {
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
std::string TypeOfTarget(const pybind11::array &y) {
    pybind11::print("WARNING:classicML: `ops.cc_type_of_target` 已经被弃用,"
                    " 它将在未来的正式版本中被移除, 请使用 `ops.cc_type_of_target_v2`.");
    return "unknown";
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 相比于`ops::TypeOfTarget`可以处理单字符的输入数据.
std::string TypeOfTarget_V2(const pybind11::array &y) {
    if ((std::string) pybind11::str(y.dtype()) == "object" or y.ndim() > 2) {
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

// 显式实例化.
template std::tuple<matrix32, matrix32> BootstrapSampling1(const matrix32 &x,
                                                           const matrix32 &y,
                                                           std::optional<uint32> seed);
template std::tuple<matrix64, matrix64> BootstrapSampling1(const matrix64 &x,
                                                           const matrix64 &y,
                                                           std::optional<uint32> seed);
template std::tuple<matrix32, matrix32i> BootstrapSampling1(const matrix32 &x,
                                                            const matrix32i &y,
                                                            std::optional<uint32> seed);
template std::tuple<matrix64, matrix64i> BootstrapSampling1(const matrix64 &x,
                                                            const matrix64i &y,
                                                            std::optional<uint32> seed);
template matrix32 BootstrapSampling2(const matrix32 &x, const pybind11::object &y, std::optional<uint32> seed);
template matrix64 BootstrapSampling2(const matrix64 &x, const pybind11::object &y, std::optional<uint32> seed);

template matrix32 CalculateError<matrix32, vector32, array32>
        (const matrix32 &x,
         const vector32 &y,
         const uint32 &i,
         const pybind11::object &kernel,
         const vector32 &alphas,
         const vector32 &non_zero_alphas,
         const matrix32 &b);
template matrix64 CalculateError<matrix64, vector64, array64>
        (const matrix64 &x,
         const vector64 &y,
         const uint32 &i,
         const pybind11::object &kernel,
         const vector64 &alphas,
         const vector64 &non_zero_alphas,
         const matrix64 &b);

template matrix32 CalculateEuclideanDistance(const matrix32 &x0, const matrix32 &x1);
template matrix64 CalculateEuclideanDistance(const matrix64 &x0, const matrix64 &x1);

template np_float32 ClipAlpha(np_float32 &alpha, np_float32 &low, np_float32 &high);
template np_float64 ClipAlpha(np_float64 &alpha, np_float64 &low, np_float64 &high);
template np_float32 ClipAlpha(float32 &alpha, float32 &low, float32 &high);

template row_vector_bool CompareDifferences(const matrix32 &x0, const matrix32 &x1, np_float32 &tol);
template row_vector_bool CompareDifferences(const matrix64 &x0, const matrix64 &x1, np_float64 &tol);
template row_vector_bool CompareDifferences(const matrix32 &x0, const matrix32 &x1, float32 &tol);

template row_vector32i GetCluster(const matrix32 &distances);
template row_vector32i GetCluster(const matrix64 &distances);

template np_float32 GetConditionalProbability(np_uint32 &samples_on_attribute,
                                              np_uint32 &samples_in_category,
                                              np_uint32 &num_of_categories,
                                              const bool &smoothing);
template np_float64 GetConditionalProbability(np_uint64 &samples_on_attribute,
                                              np_uint64 &samples_in_category,
                                              np_uint64 &num_of_categories,
                                              const bool &smoothing);
template np_float32 GetConditionalProbability(uint32 &samples_on_attribute,
                                              uint32 &samples_in_category,
                                              uint32 &num_of_categories,
                                              const bool &smoothing);

template np_float32 GetDependentPriorProbability(np_uint32 &samples_on_attribute_in_category,
                                                 np_uint32 &number_of_sample,
                                                 np_uint32 &values_on_attribute,
                                                 const bool &smoothing);
template np_float64 GetDependentPriorProbability(np_uint64 &samples_on_attribute_in_category,
                                                 np_uint64 &number_of_sample,
                                                 np_uint64 &values_on_attribute,
                                                 const bool &smoothing);
template np_float32 GetDependentPriorProbability(uint32 &samples_on_attribute_in_category,
                                                 uint32 &number_of_sample,
                                                 uint32 &values_on_attribute,
                                                 const bool &smoothing);

template std::tuple<np_float32, np_float32> GetPriorProbability
        (const uint32 &number_of_sample, const row_vector32i &y, const bool &smoothing);
template std::tuple<np_float64, np_float64> GetPriorProbability
        (const uint32 &number_of_sample, const row_vector64i &y, const bool &smoothing);

template np_float32 GetProbabilityDensity(np_float32 &sample, np_float32 &mean, np_float32 &var);
template np_float64 GetProbabilityDensity(np_float64 &sample, np_float64 &mean, np_float64 &var);
template np_float32 GetProbabilityDensity(float32 &sample, float32 &mean, float32 &var);

template matrix32 GetW_V2(const matrix32 &S_w, const matrix32 &mu_0, const matrix32 &mu_1);
template matrix64 GetW_V2(const matrix64 &S_w, const matrix64 &mu_0, const matrix64 &mu_1);

template matrix32 GetWithinClassScatterMatrix(const matrix32 &X_0,
                                              const matrix32 &X_1,
                                              const matrix32 &mu_0,
                                              const matrix32 &mu_1);
template matrix64 GetWithinClassScatterMatrix(const matrix64 &X_0,
                                              const matrix64 &X_1,
                                              const matrix64 &mu_0,
                                              const matrix64 &mu_1);

template std::tuple<uint32, np_float32> SelectSecondAlpha(np_float32 &error,
                                                          const row_vector32f &error_cache,
                                                          const row_vector32f &non_bound_alphas);
template std::tuple<uint32, np_float64> SelectSecondAlpha(np_float64 &error,
                                                          const row_vector64f &error_cache,
                                                          const row_vector64f &non_bound_alphas);
template std::tuple<uint32, float32> SelectSecondAlpha(float32 &error,
                                                       const row_vector32f &error_cache,
                                                       const row_vector32f &non_bound_alphas);
} // namespace ops