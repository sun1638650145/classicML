import numpy as np
from .kernel import linear_kernel, polynomial_kernel, rbf_kernel


def kappa_xi_x(kernel, x_i, x_j, degree, gamma):
    """选择核函数"""
    if kernel == 'linear':
        kappa_i = linear_kernel(x_i, x_j)
    elif kernel == 'poly':
        kappa_i = polynomial_kernel(x_i, x_j, degree)
    elif kernel == 'rbf':
        kappa_i = rbf_kernel(x_i, x_j, gamma)

    return kappa_i


def calc_error(x, y, i, b, alphas, non_zero_alpha, kernel, degree, gamma):
    """计算损失"""
    x_i = x[[i], :]
    y_i = y[i, 0]
    if non_zero_alpha.any():
        valid_x = x[non_zero_alpha]
        valid_y = y[non_zero_alpha]
        valid_alphas = alphas[non_zero_alpha]
        kappa_x_i = kappa_xi_x(kernel, valid_x, x_i, degree, gamma)

        gx = np.dot((valid_alphas.reshape(-1, 1) * valid_y).T, kappa_x_i) + b
    else:
        gx = b

    error = gx - y_i

    return np.squeeze(error)


def select_second_alpha(error, error_cache, non_bound_alpha):
    """第二个变量的选择使得目标函数数值增长最大"""

    non_bound_index = non_bound_alpha.nonzero()[0]
    delta_E = np.abs(error - error_cache[non_bound_index])

    index_alpha_j = non_bound_index[np.argmax(delta_E)]
    error_alpha_j = error_cache[index_alpha_j]

    return index_alpha_j, error_alpha_j


def kappa_xi_xj(kernel, x, i, j, degree, gamma):
    """选择核函数"""
    x_i = x[[i], :]
    x_j = x[[j], :]

    if kernel == 'linear':
        kappa_i = linear_kernel(x_i, x_j)
    elif kernel == 'poly':
        kappa_i = polynomial_kernel(x_i, x_j, degree)
    elif kernel == 'rbf':
        kappa_i = rbf_kernel(x_i, x_j, gamma)

    return np.squeeze(kappa_i)


def clip_alpha(alpha, low, high):
    """修剪alpha"""
    if alpha > high:
        alpha = high
    elif alpha < low:
        alpha = low

    return alpha


def update_error_cache(svc, x, y):
    """更新误差缓存"""
    for sample in svc.non_bound_alpha.nonzero()[0]:
        svc.error_cache[sample] = calc_error(x, y, sample, svc.b, svc.alphas, svc.non_zero_alpha, svc.kernel, svc.degree, svc.gamma)


def update_alpha_array(svc, alpha, index):
    """更新non_zero_alpha和non_bound_alpha"""
    svc.non_zero_alpha[index] = (alpha > 0)
    svc.non_bound_alpha[index] = int(0 < alpha < svc.C)


def update_alpha(svc, x, y, i, j, error_i, error_j):
    """更新alpha"""
    # 失败情况1
    if i == j:
        return False

    alpha_i_old = svc.alphas[i].copy()
    alpha_j_old = svc.alphas[j].copy()
    y_i = y[i, 0]
    y_j = y[j, 0]

    if y_i != y_j:
        low = max(0, alpha_j_old - alpha_i_old)
        high = min(svc.C, svc.C + alpha_j_old - alpha_i_old)
    else:
        low = max(0, alpha_j_old + alpha_i_old - svc.C)
        high = min(svc.C, alpha_j_old + alpha_i_old)
    # 失败情况2
    if low == high:
        return False

    kappa_ii = kappa_xi_xj(svc.kernel, x, i, i, svc.degree, svc.gamma)
    kappa_ij = kappa_xi_xj(svc.kernel, x, i, j, svc.degree, svc.gamma)
    kappa_jj = kappa_xi_xj(svc.kernel, x, j, j, svc.degree, svc.gamma)
    eta = kappa_ii + kappa_jj - 2 * kappa_ij
    # 失败情况3
    if eta <= 0:
        return False

    alpha_j_new = alpha_j_old + y_j * (error_i - error_j) / eta
    alpha_j_new = clip_alpha(alpha_j_new, low, high)
    # 失败情况4
    if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
        return False

    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
    b_j = svc.b - error_j - y_j * kappa_jj * (alpha_j_new - alpha_j_old) - y_i * kappa_ij * (alpha_i_new - alpha_i_old)
    b_i = svc.b - error_i - y_i * kappa_ij * (alpha_j_new - alpha_j_old) - y_i * kappa_ii * (alpha_i_new - alpha_i_old)

    if 0 < alpha_j_new < svc.C:
        svc.b = b_j
    elif 0 < alpha_i_new < svc.C:
        svc.b = b_i
    else:
        svc.b = (b_i + b_j) / 2

    svc.alphas[i] = alpha_i_new
    svc.alphas[j] = alpha_j_new

    update_error_cache(svc, x, y)
    update_alpha_array(svc, alpha_i_new, i)
    update_alpha_array(svc, alpha_j_new, j)

    return True