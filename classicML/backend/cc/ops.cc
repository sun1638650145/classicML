//
//  ops.cc
//  ops
//
//  Created by 孙瑞琦 on 2020/10/10.
//
//

#include "ops.h"

// 返回一个矩阵, 计算向量减法, 输入为矩阵和行向量(矩阵和行向量的列数必须相同).
Eigen::MatrixXd Sub(const Eigen::MatrixXd &matrix, const Eigen::RowVectorXd &vector) {
    if (matrix.cols() != vector.cols()) {
        throw pybind11::value_error("列数不同, 无法操作");
    }

    Eigen::MatrixXd new_matrix(matrix.rows(), matrix.cols());
    for (int row = 0; row < matrix.rows(); row ++) {
        new_matrix.row(row) = matrix.row(row) - vector;
    }

    return new_matrix;
}

// 返回KKT条件的违背值;
// 输入特征数据, 标签, 要计算的样本下标, 支持向量分类器使用的核函数, 全部拉格朗日乘子, 非零拉格朗日乘子和偏置项.
// TODO(Steve R. Sun, tag:performance): 在CC中使用Python实现的核函数实际性能和Python没差别, 但是由于其他地方依旧使用的是CC代码, 还是会有明显的性能提高.
Eigen::MatrixXd CalculateError(const Eigen::MatrixXd &x,
                               const Eigen::MatrixXd &y, // 列向量
                               const int &i,
                               const pybind11::object &kernel,
                               const Eigen::MatrixXd &alphas,  // 列向量
                               const Eigen::VectorXd &non_zero_alphas,
                               const Eigen::MatrixXd &b) {
    Eigen::MatrixXd x_i = x.row(i);
    Eigen::MatrixXd y_i = y.row(i);

    // 拉格朗日乘子全是零的时候, 每个样本都不会对结果产生影响.
    Eigen::MatrixXd fx = b;
    // 如果有非零元素, 提取全部合格的标签和对应的拉格朗日乘子.
    if (non_zero_alphas.any()) {
        Eigen::MatrixXd valid_x = GetNonZeroSubMatrix(x, non_zero_alphas);
        Eigen::ArrayXd valid_y = GetNonZeroSubMatrix(y, non_zero_alphas);
        Eigen::MatrixXd valid_alphas = GetNonZeroSubMatrix(alphas, non_zero_alphas);

        // 调用核函数的__call__方法.
        pybind11::object py_kappa = kernel(valid_x, x_i);
        Eigen::MatrixXd kappa = py_kappa.cast<Eigen::MatrixXd>();

        // 这里是Hadamard积, 故临时需要使用ArrayXd.
        Eigen::ArrayXd temp = Reshape(valid_alphas, -1, 1);
        temp = temp * valid_y;
        Eigen::MatrixXd matrix_temp = (Eigen::MatrixXd)temp.transpose();

        fx = matrix_temp * kappa.transpose() + b;
    }

    Eigen::MatrixXd error = fx - y_i;

    return error;
}

// 返回修剪后的拉格朗日乘子, 输入拉格朗日乘子的下界和上界.
Eigen::ArrayXd ClipAlpha(const double &alpha, const double &low, const double &high) {
    Eigen::MatrixXd clipped_alpha(1, 1);
    clipped_alpha(0, 0) = alpha;

    if (alpha > high) {
        clipped_alpha(0, 0) = high;
    } else if (alpha < low) {
        clipped_alpha(0, 0) = low;
    }

    return clipped_alpha;
}

// 获取类条件概率, 输入某个属性值的样本总数, 某个类别的样本总数, 类别的数量和是否使用平滑.
double GetConditionalProbability(const double &samples_on_attribute,
                                 const int &samples_in_category,
                                 const int &num_of_categories,
                                 const bool &smoothing) {
    if (smoothing) {
        return (samples_on_attribute + 1) / (samples_in_category + num_of_categories);
    } else {
        return samples_on_attribute / samples_in_category;
    }
}

// 获取有依赖的类先验概率, 输入类别为c的属性i上取值为xi的样本, 样本的总数, 特征数据和是否使用平滑.
double GetDependentPriorProbability(const int &samples_on_attribute_in_category,
                                    const int &number_of_sample,
                                    const int &values_on_attribute,
                                    const bool &smoothing) {
    double probability;

    if (smoothing) {
        probability = (double)(samples_on_attribute_in_category + 1) / (number_of_sample + 2 * values_on_attribute);
    } else {
        probability = (double)samples_on_attribute_in_category / number_of_sample;
    }

    return probability;
}

// 获取类先验概率, 输入特征数据, 标签和是否使用平滑.
std::tuple<double, double> GetPriorProbability(const int &number_of_sample,
                                               const Eigen::RowVectorXd &y,
                                               const bool &smoothing) {
    // 遍历获得反例的个数.
    double num_of_negative_sample = 0.0;
    for (int i = 0; i < y.size(); i ++) {
        if (y[i] == 0) {
            num_of_negative_sample += 1;
        }
    }

    if (smoothing) {
        double p_0 = (num_of_negative_sample + 1) / (number_of_sample + 2);
        std::tuple<double, double> probability(p_0, 1 - p_0);

        return probability;
    } else {
        double p_0 = num_of_negative_sample / number_of_sample;
        std::tuple<double, double> probability(p_0, 1 - p_0);

        return probability;
    }
}


// 获取概率密度, 输入样本的取值, 样本在某个属性的上的均值和样本在某个属性上的方差.
double GetProbabilityDensity(const double &sample,
                             const double &mean,
                             const double &var) {
    double probability =  1 / (sqrt(2 * M_PI) * var) * exp(-(pow((sample - mean), 2) / (2 * pow(var, 2))));

    // probability有可能为零, 导致取对数会有异常, 因此选择一个常小数.
    if (probability == 0) {
        probability = 1e-36;
    }

    return probability;
}

// 返回投影向量, 输入为类内散度矩阵和反正例的均值向量.
Eigen::MatrixXd GetW(const Eigen::MatrixXd &S_w, const Eigen::MatrixXd &mu_0, const Eigen::MatrixXd &mu_1) {
    // TODO(Steve R. Sun, tag:bug): 应该使用奇异值分解的方式求解S_w的逆矩阵.
    // 公式(数学公式难以表示, 使用latex语法): w = (S_w)^{-1}(\mu_0 - \mu_1)
    Eigen::MatrixXd S_w_inv = S_w.inverse();
    Eigen::MatrixXd mu_t = (mu_0 - mu_1).transpose();

    Eigen::MatrixXd w = S_w_inv * mu_t;

    return Reshape(w, 1, -1);
}

// 返回类内散度矩阵, 输入为反正例的集合矩阵和反正例的均值向量.
Eigen::MatrixXd GetWithinClassScatterMatrix(const Eigen::MatrixXd &X_0,
                                            const Eigen::MatrixXd &X_1,
                                            const Eigen::MatrixXd &mu_0,
                                            const Eigen::MatrixXd &mu_1) {
    // 公式(数学公式难以表示, 使用latex语法):
    // S_w = \sum_0 + \sum_1
    // \sum_i = \sum_{x \in X_i}(x - \mu_0)(x - \mu_1)^T
    Eigen::MatrixXd S_0 = (Sub(X_0, mu_0)).transpose() * (Sub(X_0, mu_0));
    Eigen::MatrixXd S_1 = (Sub(X_1, mu_1)).transpose() * (Sub(X_1, mu_1));

    Eigen::MatrixXd S_w = S_0 + S_1;

    return S_w;
}

// 返回第二个拉格朗日乘子的下标和违背值组成的元组, 输入KKT条件的违背值, KKT条件的违背值缓存和非边界拉格朗日乘子.
std::tuple<int, double> SelectSecondAlpha(const double &error,
                                          const Eigen::RowVectorXd &error_cache,
                                          const Eigen::RowVectorXd &non_bound_alphas) {
    std::vector<int> non_bound_index = NonZero(non_bound_alphas);

    int index_alpha = 0;
    double error_alpha = error_cache[index_alpha];
    double delta_e = abs(error - error_cache[non_bound_index[0]]);

    // 选取最大间隔的拉格朗日乘子对应的下标和违背值.
    for (int i : non_bound_index) {
        double temp = abs(error - error_cache[i]);
        if (temp > delta_e) {
            delta_e = temp;
            index_alpha = i;
            error_alpha = error_cache[index_alpha];
        }
    }

    std::tuple<int, double> alpha_tuple(index_alpha, error_alpha);

    return alpha_tuple;
}

// 返回输入数据的数据类型的字符串, 输入为待测试数据.
// 只处理float64的输入数据.
std::string TypeOfTarget(const Eigen::MatrixXd &y) {
    bool any = true;
    std::set<double> buffer;

    // 任何一个元素取整不等于它本身的就是连续值.
    for (int row = 0; row < y.rows(); row ++) {
        for (int col = 0; col < y.cols(); col ++) {
            buffer.insert(y(row, col));
            if (y(row, col) != (int)y(row, col)) {
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
std::string TypeOfTarget(const Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> &y) {
    // 取唯一值统计元素个数.
    std::set<double> buffer;
    for (int row = 0; row < y.rows(); row ++) {
        for (int col = 0; col < y.cols(); col ++) {
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
std::string TypeOfTarget(const pybind11::array &y) {
    return "unknown";
}