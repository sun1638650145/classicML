"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.
"""
import os
import numpy as np

from classicML import _cml_precision
from classicML.backend import kernels

from classicML.backend.cc.ops import cc_bootstrap_sampling
from classicML.backend.cc.ops import ConvexHull as CcConvexHull
from classicML.backend.cc.ops import cc_calculate_centroids
from classicML.backend.cc.ops import cc_calculate_error
from classicML.backend.cc.ops import cc_calculate_euclidean_distance
from classicML.backend.cc.ops import cc_calculate_means
from classicML.backend.cc.ops import cc_clip_alpha
from classicML.backend.cc.ops import cc_compare_differences
from classicML.backend.cc.ops import cc_get_cluster
from classicML.backend.cc.ops import cc_get_conditional_probability
from classicML.backend.cc.ops import cc_get_dependent_prior_probability
from classicML.backend.cc.ops import cc_get_prior_probability
from classicML.backend.cc.ops import cc_get_probability_density
from classicML.backend.cc.ops import cc_get_w_v2 as cc_get_w
from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix
from classicML.backend.cc.ops import cc_init_centroids
from classicML.backend.cc.ops import cc_select_second_alpha
from classicML.backend.cc.ops import cc_type_of_target_v2 as cc_type_of_target

from classicML.backend.python.ops import bootstrap_sampling
from classicML.backend.python.ops import ConvexHull
from classicML.backend.python.ops import calculate_centroids
from classicML.backend.python.ops import calculate_error
from classicML.backend.python.ops import calculate_euclidean_distance
from classicML.backend.python.ops import calculate_means
from classicML.backend.python.ops import clip_alpha
from classicML.backend.python.ops import compare_differences
from classicML.backend.python.ops import get_cluster
from classicML.backend.python.ops import get_conditional_probability
from classicML.backend.python.ops import get_dependent_prior_probability
from classicML.backend.python.ops import get_prior_probability
from classicML.backend.python.ops import get_probability_density
from classicML.backend.python.ops import get_w_v2 as get_w
from classicML.backend.python.ops import get_within_class_scatter_matrix
from classicML.backend.python.ops import init_centroids
from classicML.backend.python.ops import select_second_alpha
from classicML.backend.python.ops import type_of_target

if os.environ['CLASSICML_PRECISION'] == '32-bit':
    THRESHOLD = 1e-7
elif os.environ['CLASSICML_PRECISION'] == '64-bit':
    THRESHOLD = 1e-15


class TestBootstrapSampling(object):
    def test_x_y_seed(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 1)
        seed = 2022

        cc_answer = cc_bootstrap_sampling(x, y, seed)
        py_answer = bootstrap_sampling(x, y, seed)
        assert cc_answer[0].shape == py_answer[0].shape  # x
        assert cc_answer[1].shape == py_answer[1].shape  # y

    def test_x_y(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 1)

        cc_answer = cc_bootstrap_sampling(x, y)
        py_answer = bootstrap_sampling(x, y)
        assert cc_answer[0].shape == py_answer[0].shape  # x
        assert cc_answer[1].shape == py_answer[1].shape  # y

    def test_y(self):
        # 数据为随机产生, 不具有任何实际意义.
        y = np.random.rand(5, 2)

        cc_answer = cc_bootstrap_sampling(x=y)
        py_answer = bootstrap_sampling(x=y)
        assert cc_answer[0].shape == py_answer[0].shape  # y

    def test_x_seed(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(5, 2)
        seed = 2022

        cc_answer = cc_bootstrap_sampling(x, seed=seed)
        py_answer = bootstrap_sampling(x, seed=seed)
        assert cc_answer[0].shape == py_answer[0].shape  # x


class TestConvexHull(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        points = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(10)]

        cc_answer = CcConvexHull(points).hull
        py_answer = ConvexHull(points).hull

        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestCalculateCentroids(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(10, 5)
        clusters = np.random.randint(0, 5, 10)

        cc_answer = cc_calculate_centroids(x, clusters)
        py_answer = calculate_centroids(x, clusters)

        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestCalculateError(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 1).astype(int)
        i = 3
        kernel = kernels.Linear()
        alphas = [4]
        non_zero_alphas = np.random.rand(5, 1).astype(int)
        b = [0]

        cc_answer = cc_calculate_error(x, y, i, kernel, alphas, non_zero_alphas, b)
        py_answer = calculate_error(x, y, i, kernel, alphas, non_zero_alphas, b)
        assert abs(cc_answer - py_answer) <= THRESHOLD


class TestCalculateEuclideanDistance(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        x0 = np.random.random(size=[30, 3])
        x1 = np.random.random(size=[3, 3])

        cc_answer = cc_calculate_euclidean_distance(x0, x1)
        py_answer = calculate_euclidean_distance(x0, x1)

        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestCalculateMeans(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.random(size=[30, 2])
        gamma = np.random.random(size=[30, 3])

        cc_answer = cc_calculate_means(x, gamma)
        py_answer = calculate_means(x, gamma)

        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestClipAlpha(object):
    def test_in_interval(self):
        alpha, low, high = 4.5, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert abs(cc_answer - py_answer) <= THRESHOLD and (py_answer == alpha)

    def test_outside_left_interval(self):
        alpha, low, high = 3, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert abs(cc_answer - py_answer) <= THRESHOLD and (py_answer == low)

    def test_outside_right_interval(self):
        alpha, low, high = 6, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert abs(cc_answer - py_answer) <= THRESHOLD and (py_answer == high)


class TestCompareDifferences(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.

        # matrix.
        centroids_matrix = np.random.random(size=[30, 3])
        new_centroids_matrix = np.random.random(size=[30, 3])
        # tensor.
        centroids_tensor = np.random.random(size=[30, 3, 2])
        new_centroids_tensor = np.random.random(size=[30, 3, 2])
        tol = np.random.random()

        cc_answer_matrix = cc_compare_differences(centroids_matrix, new_centroids_matrix, tol)
        py_answer_matrix = compare_differences(centroids_matrix, new_centroids_matrix, tol)
        cc_answer_tensor = cc_compare_differences(centroids_tensor, new_centroids_tensor, tol)
        py_answer_tensor = compare_differences(centroids_tensor, new_centroids_tensor, tol)

        assert cc_answer_matrix.all() == py_answer_matrix.all()
        assert cc_answer_tensor.all() == py_answer_tensor.all()


class TestGetCluster(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        distances = np.random.random(size=[30, 3])

        cc_answer = cc_get_cluster(distances)
        py_answer = get_cluster(distances)

        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestGetConditionalProbability(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        samples_on_attribute = np.random.randint(5, 10)
        samples_in_category = np.random.randint(1, 3)
        num_of_categories = np.random.randint(10, 20)

        cc_answer_smoothing = cc_get_conditional_probability(samples_on_attribute,
                                                             samples_in_category,
                                                             num_of_categories,
                                                             True)
        py_answer_smoothing = get_conditional_probability(samples_on_attribute,
                                                          samples_in_category,
                                                          num_of_categories,
                                                          True)

        cc_answer = cc_get_conditional_probability(samples_on_attribute, samples_in_category, num_of_categories, False)
        py_answer = get_conditional_probability(samples_on_attribute, samples_in_category, num_of_categories, False)

        assert abs(cc_answer_smoothing - py_answer_smoothing) <= THRESHOLD
        assert abs(cc_answer - py_answer) <= THRESHOLD


class TestGetDependentPriorProbability(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        samples_on_attribute_in_category = np.random.randint(5, 10)
        number_of_sample = np.random.randint(20, 30)
        values_on_attribute = np.random.randint(3, 5)

        cc_answer_smoothing = cc_get_dependent_prior_probability(samples_on_attribute_in_category,
                                                                 number_of_sample,
                                                                 values_on_attribute,
                                                                 True)
        py_answer_smoothing = get_dependent_prior_probability(samples_on_attribute_in_category,
                                                              number_of_sample,
                                                              values_on_attribute,
                                                              True)

        cc_answer = cc_get_conditional_probability(samples_on_attribute_in_category,
                                                   number_of_sample,
                                                   values_on_attribute,
                                                   False)
        py_answer = get_conditional_probability(samples_on_attribute_in_category,
                                                number_of_sample,
                                                values_on_attribute,
                                                False)

        assert abs(cc_answer_smoothing - py_answer_smoothing) <= THRESHOLD
        assert abs(cc_answer - py_answer) <= THRESHOLD


class TestGetPriorProbability(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        number_of_sample = np.random.randint(10, 15)
        y = np.asarray([1, 1, 0, 0, 1, 0, 1, 0, 1], dtype=_cml_precision.int)

        # 概率和为1, 所以只验证其中一个即可.
        cc_answer_smoothing_p0, _ = cc_get_prior_probability(number_of_sample, y, True)
        py_answer_smoothing_p0, _ = get_prior_probability(number_of_sample, y, True)

        cc_answer_p0, _ = cc_get_prior_probability(number_of_sample, y, False)
        py_answer_p0, _ = np.asarray(get_prior_probability(number_of_sample, y, False))

        assert abs(cc_answer_smoothing_p0 - py_answer_smoothing_p0) <= THRESHOLD
        assert abs(cc_answer_p0 - py_answer_p0) <= THRESHOLD


class TestGetProbabilityDensity(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        sample = _cml_precision.float(np.random.rand())
        mean = _cml_precision.float(np.random.rand())
        var = _cml_precision.float(np.random.rand())

        cc_answer = cc_get_probability_density(sample, mean, var)
        py_answer = get_probability_density(sample, mean, var)

        assert abs(cc_answer - py_answer) <= THRESHOLD


class TestGetW(object):
    def test_answer(self):
        S_w = np.asarray([[1, 2], [3, 4]], dtype=_cml_precision.float)
        mu_0 = np.asarray([[1, 1]], dtype=_cml_precision.float)
        mu_1 = np.asarray([[1, 1]], dtype=_cml_precision.float)

        cc_answer = cc_get_w(S_w, mu_0, mu_1)
        py_answer = get_w(S_w, mu_0, mu_1)

        assert cc_answer.all() == py_answer.all()


class TestGetWithinClassScatterMatrix(object):
    def test_answer(self):
        X_0 = np.asarray([[1, 2], [3, 4]], dtype=_cml_precision.float)
        X_1 = np.asarray([[3, 4], [1, 2]], dtype=_cml_precision.float)
        mu_0 = np.asarray([[1, 2]], dtype=_cml_precision.float)
        mu_1 = np.asarray([[3, 4]], dtype=_cml_precision.float)

        cc_answer = cc_get_within_class_scatter_matrix(X_0, X_1, mu_0, mu_1)
        py_answer = get_within_class_scatter_matrix(X_0, X_1, mu_0, mu_1)

        assert cc_answer.all() == py_answer.all()


class TestInitCentroids(object):
    def test_str(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(10, 3)
        n_clusters = 3
        init = 'random'

        cc_answer = cc_init_centroids(x, n_clusters, init)
        py_answer = init_centroids(x, n_clusters, init)

        assert cc_answer.shape == py_answer.shape

    def test_indices(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(10, 3)
        n_clusters = 3
        init = np.random.randint(0, 10, size=n_clusters)

        cc_answer = cc_init_centroids(x, n_clusters, init)
        py_answer = init_centroids(x, n_clusters, init)
        assert abs(cc_answer - py_answer).max() <= THRESHOLD

    def test_specified_value(self):
        # 数据为随机产生, 不具有任何实际意义.
        x = np.random.rand(10, 3)
        n_clusters = 3
        init = np.random.choice(len(x), n_clusters)

        cc_answer = cc_init_centroids(x, n_clusters, init)
        py_answer = init_centroids(x, n_clusters, init)
        assert abs(cc_answer - py_answer).max() <= THRESHOLD


class TestSelectSecondAlpha(object):
    def test_answer(self):
        error = 0.5
        error_cache = np.asarray([0.1, 0.2, 0.3], dtype=_cml_precision.float)
        non_bound_alphas = np.asarray([1., 2., 3.], dtype=_cml_precision.float)

        cc_answer = cc_select_second_alpha(error, error_cache, non_bound_alphas)
        py_answer = select_second_alpha(error, error_cache, non_bound_alphas)

        assert cc_answer == py_answer


class TestTypeOfTarget(object):
    def test_binary(self):
        y = np.asarray([1.0, 0.0, 1.0, 1.0])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'binary')

    def test_continuous(self):
        y = np.asarray([1.0, 1.1, 1.2])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'continuous')

    def test_multiclass(self):
        y = np.asarray([1.0, 2.0, 3.0])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'multiclass')

    def test_multilabel(self):
        y = np.asarray([[1, 2], [0, 1], [0, 1], [0, 2]])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'multilabel')

    def test_unknown(self):
        y = np.asarray(['a', 1, [2, 'b']], dtype='object')

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'unknown')

    def test_char_binary(self):
        y = np.asarray(['a', 'b', 'a'])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'binary')

    def test_char_multiclass(self):
        y = np.asarray([['a'], ['b'], ['c']])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'multiclass')

    def test_char_multilabel(self):
        y = np.asarray([['a', 'b'], ['a', 'c'], ['b', 'c'], ['a', 'b']])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'multilabel')
