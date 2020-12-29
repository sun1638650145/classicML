import os

from classicML import CLASSICML_LOGGER

from classicML.backend.python import activations
from classicML.backend.python import callbacks
from classicML.backend.python import initializers
from classicML.backend.python import io
from classicML.backend.python import kernels
from classicML.backend.python import losses
if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.metrics import __version__
    from classicML.backend.cc import metrics
    CLASSICML_LOGGER.info('后端版本是: {}'.format(__version__))
else:
    from classicML.backend.python import metrics

from classicML.backend.python import optimizers
from classicML.backend.python import tree

if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.ops import cc_calculate_error
    from classicML.backend.cc.ops import cc_clip_alpha
    from classicML.backend.cc.ops import cc_get_conditional_probability
    from classicML.backend.cc.ops import cc_get_dependent_prior_probability
    from classicML.backend.cc.ops import cc_get_prior_probability
    from classicML.backend.cc.ops import cc_get_probability_density
    from classicML.backend.cc.ops import cc_get_w
    from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix
    from classicML.backend.cc.ops import cc_select_second_alpha
    from classicML.backend.cc.ops import cc_type_of_target

    from classicML.backend.cc.ops import cc_calculate_error as calculate_error
    from classicML.backend.cc.ops import cc_clip_alpha as clip_alpha
    from classicML.backend.cc.ops import cc_get_conditional_probability as get_conditional_probability
    from classicML.backend.cc.ops import cc_get_dependent_prior_probability as get_dependent_prior_probability
    from classicML.backend.cc.ops import cc_get_prior_probability as get_prior_probability
    from classicML.backend.cc.ops import cc_get_probability_density as get_probability_density
    from classicML.backend.cc.ops import cc_get_w as get_w
    from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix as get_within_class_scatter_matrix
    from classicML.backend.cc.ops import cc_select_second_alpha as select_second_alpha
    from classicML.backend.cc.ops import cc_type_of_target as type_of_target

    from classicML.backend.cc.ops import __version__
    CLASSICML_LOGGER.info('后端版本是: {}'.format(__version__))
else:
    from classicML.backend.python.ops import calculate_error
    from classicML.backend.python.ops import clip_alpha
    from classicML.backend.python.ops import get_conditional_probability
    from classicML.backend.python.ops import get_dependent_prior_probability
    from classicML.backend.python.ops import get_prior_probability
    from classicML.backend.python.ops import get_probability_density
    from classicML.backend.python.ops import get_w
    from classicML.backend.python.ops import get_within_class_scatter_matrix
    from classicML.backend.python.ops import select_second_alpha
    from classicML.backend.python.ops import type_of_target

from classicML.backend.training import get_initializer
from classicML.backend.training import get_kernel
from classicML.backend.training import get_loss
from classicML.backend.training import get_metric
from classicML.backend.training import get_optimizer
from classicML.backend.training import get_pruner