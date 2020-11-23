"""
    测试cc后端和默认python后端的运行结果是否一致.
    使用的随机数据采取了一些限制以更好满足测试, 但仍然不具有实际意义.
"""
import numpy as np

from classicML.backend import kernels

from classicML.backend.cc.ops import cc_calculate_error
from classicML.backend.cc.ops import cc_clip_alpha
from classicML.backend.cc.ops import cc_get_conditional_probability
from classicML.backend.cc.ops import cc_get_prior_probability
from classicML.backend.cc.ops import cc_get_probability_density
from classicML.backend.cc.ops import cc_get_w
from classicML.backend.cc.ops import cc_type_of_target

from classicML.backend.python.ops import calculate_error
from classicML.backend.python.ops import clip_alpha
from classicML.backend.python.ops import get_conditional_probability
from classicML.backend.python.ops import get_prior_probability
from classicML.backend.python.ops import get_probability_density
from classicML.backend.python.ops import get_w
from classicML.backend.python.ops import type_of_target


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
        assert cc_answer == py_answer


class TestClipAlpha(object):
    def test_in_interval(self):
        alpha, low, high = 4.5, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert (cc_answer == py_answer) and (py_answer == alpha)

    def test_outside_left_interval(self):
        alpha, low, high = 3, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert (cc_answer == py_answer) and (py_answer == low)

    def test_outside_right_interval(self):
        alpha, low, high = 6, 4, 5

        cc_answer = cc_clip_alpha(alpha, low, high)
        py_answer = clip_alpha(alpha, low, high)
        assert (cc_answer == py_answer) and (py_answer == high)


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

        assert cc_answer_smoothing == py_answer_smoothing
        assert cc_answer == py_answer


class TestGetPriorProbability(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        number_of_sample = np.random.randint(10, 15)
        y = np.asarray([1, 1, 0, 0, 1, 0, 1, 0, 1])

        cc_answer_smoothing = cc_get_prior_probability(number_of_sample, y, True)
        py_answer_smoothing = get_prior_probability(number_of_sample, y, True)

        cc_answer = cc_get_prior_probability(number_of_sample, y, False)
        py_answer = get_prior_probability(number_of_sample, y, False)

        assert cc_answer_smoothing == py_answer_smoothing
        assert cc_answer == py_answer


class TestGetProbabilityDensity(object):
    def test_answer(self):
        # 数据为随机产生, 不具有任何实际意义.
        sample = np.random.rand()
        mean = np.random.rand()
        var = np.random.rand()

        cc_answer = cc_get_probability_density(sample, mean, var)
        py_answer = get_probability_density(sample, mean, var)

        assert cc_answer == py_answer


class TestGetW(object):
    def test_answer(self):
        S_w = np.asmatrix([[1, 2], [3, 4]])
        mu_0 = np.asarray([[1, 2]])
        mu_1 = np.asarray([[3, 4]])

        cc_answer = cc_get_w(S_w, mu_0, mu_1)
        py_answer = get_w(S_w, mu_0, mu_1)

        assert cc_answer.all() == py_answer.all()


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
        y = np.asarray(['a', 1, [2, 'b']])

        cc_answer = cc_type_of_target(y)
        py_answer = type_of_target(y)

        assert (cc_answer == py_answer) and (py_answer == 'unknown')