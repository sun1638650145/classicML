import os

from classicML import CLASSICML_LOGGER

# python和cc多后端模块
if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc import activations
    from classicML.backend.python import callbacks
    from classicML.backend.python import initializers
    from classicML.backend.python import io
    from classicML.backend.cc import kernels
    from classicML.backend.cc import losses
    from classicML.backend.cc import metrics
    from classicML.backend.python import optimizers
    from classicML.backend.python import tree

    from classicML.backend.cc.activations import __version__ as activations__version__
    from classicML.backend.cc.kernels import __version__ as kernels__version__
    from classicML.backend.cc.losses import __version__ as losses__version__
    from classicML.backend.cc.metrics import __version__ as metrics__version__

    CLASSICML_LOGGER.info('后端版本是: {}'.format(activations__version__))
    CLASSICML_LOGGER.info('后端版本是: {}'.format(kernels__version__))
    CLASSICML_LOGGER.info('后端版本是: {}'.format(losses__version__))
    CLASSICML_LOGGER.info('后端版本是: {}'.format(metrics__version__))
else:
    from classicML.backend.python import activations
    from classicML.backend.python import callbacks
    from classicML.backend.python import initializers
    from classicML.backend.python import io
    from classicML.backend.python import kernels
    from classicML.backend.python import losses
    from classicML.backend.python import metrics
    from classicML.backend.python import optimizers
    from classicML.backend.python import tree

# ops模块
if os.environ['CLASSICML_ENGINE'] == 'CC':
    from classicML.backend.cc.ops import cc_calculate_error
    from classicML.backend.cc.ops import cc_clip_alpha
    from classicML.backend.cc.ops import cc_get_conditional_probability
    from classicML.backend.cc.ops import cc_get_dependent_prior_probability
    from classicML.backend.cc.ops import cc_get_prior_probability
    from classicML.backend.cc.ops import cc_get_probability_density
    from classicML.backend.cc.ops import cc_get_w as cc_get_w_v1  # 正式版将移除.
    from classicML.backend.cc.ops import cc_get_w_v2
    from classicML.backend.cc.ops import cc_get_w_v2 as cc_get_w
    from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix
    from classicML.backend.cc.ops import cc_select_second_alpha
    from classicML.backend.cc.ops import cc_type_of_target

    from classicML.backend.cc.ops import cc_calculate_error as calculate_error
    from classicML.backend.cc.ops import cc_clip_alpha as clip_alpha
    from classicML.backend.cc.ops import cc_get_conditional_probability as get_conditional_probability
    from classicML.backend.cc.ops import cc_get_dependent_prior_probability as get_dependent_prior_probability
    from classicML.backend.cc.ops import cc_get_prior_probability as get_prior_probability
    from classicML.backend.cc.ops import cc_get_probability_density as get_probability_density
    from classicML.backend.cc.ops import cc_get_w_v2 as get_w
    from classicML.backend.cc.ops import cc_get_within_class_scatter_matrix as get_within_class_scatter_matrix
    from classicML.backend.cc.ops import cc_select_second_alpha as select_second_alpha
    from classicML.backend.cc.ops import cc_type_of_target as type_of_target

    from classicML.backend.cc.ops import __version__ as ops__version__
    CLASSICML_LOGGER.info('后端版本是: {}'.format(ops__version__))
else:
    from classicML.backend.python.ops import calculate_error
    from classicML.backend.python.ops import clip_alpha
    from classicML.backend.python.ops import get_conditional_probability
    from classicML.backend.python.ops import get_dependent_prior_probability
    from classicML.backend.python.ops import get_prior_probability
    from classicML.backend.python.ops import get_probability_density
    from classicML.backend.python.ops import get_w as get_w_v1  # 正式版将移除.
    from classicML.backend.python.ops import get_w_v2
    from classicML.backend.python.ops import get_w_v2 as get_w
    from classicML.backend.python.ops import get_within_class_scatter_matrix
    from classicML.backend.python.ops import select_second_alpha
    from classicML.backend.python.ops import type_of_target

from classicML.backend.training import get_initializer
from classicML.backend.training import get_kernel
from classicML.backend.training import get_loss
from classicML.backend.training import get_metric
from classicML.backend.training import get_optimizer
from classicML.backend.training import get_pruner
