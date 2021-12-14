import os
import numpy as np

from classicML import _cml_precision
from classicML.backend import kernels

from classicML.backend.cc.ops import cc_calculate_error
from classicML.backend.cc.ops import cc_clip_alpha
from classicML.backend.cc.ops import cc_get_conditional_probability
from classicML.backend.cc.ops import cc_get_dependent_prior_probability
from classicML.backend.cc.ops import cc_get_prior_probability
from classicML.backend.cc.ops import cc_get_probability_density
from classicML.backend.cc.ops import cc_get_w_v2 as cc_get_w
from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix
from classicML.backend.cc.ops import cc_select_second_alpha
from classicML.backend.cc.ops import cc_type_of_target_v2 as cc_type_of_target

from classicML.backend.python.ops import calculate_error
from classicML.backend.python.ops import clip_alpha
from classicML.backend.python.ops import get_conditional_probability
from classicML.backend.python.ops import get_dependent_prior_probability
from classicML.backend.python.ops import get_prior_probability
from classicML.backend.python.ops import get_probability_density
from classicML.backend.python.ops import get_w_v2 as get_w
from classicML.backend.python.ops import get_within_class_scatter_matrix
from classicML.backend.python.ops import select_second_alpha
from classicML.backend.python.ops import type_of_target

if os.environ['CLASSICML_PRECISION'] == '32-bit':
    THRESHOLD = 1e-7
elif os.environ['CLASSICML_PRECISION'] == '64-bit':
    THRESHOLD = 1e-15


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


class TestSelectSecondAlpha(object):
    def test_answer(self):
        error = 0.5
        error_cache = np.asarray([0.1, 0.2, 0.3])
        non_bound_alphas = np.asarray([1., 2., 3.])

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
