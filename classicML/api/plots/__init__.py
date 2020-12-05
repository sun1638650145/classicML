import os

from matplotlib import pyplot as _plt

from classicML.api.plots.bayes import plot_bayes
from classicML.api.plots.callbacks import plot_history
from classicML.api.plots.linear_model import plot_linear_discriminant_analysis
from classicML.api.plots.linear_model import plot_linear_discriminant_analysis as plot_lda
from classicML.api.plots.linear_model import plot_logistic_regression
from classicML.api.plots.support_vector_machine import plot_support_vector_classifier
from classicML.api.plots.support_vector_machine import plot_support_vector_classifier as plot_svc
from classicML.api.plots.tree import plot_tree

_plt.rcParams['font.family'] = os.environ.get('CLASSICML_FONT')