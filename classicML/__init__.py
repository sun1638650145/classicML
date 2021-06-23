"""An easy-to-use ML framework."""
__version__ = '0.6.2b1'

import os
import logging

# 配置默认环境变量
os.environ.setdefault('CLASSICML_ENGINE', 'Python')
os.environ.setdefault('CLASSICML_FONT', 'Arial Unicode MS')
# 系统logger
logging.basicConfig(level=logging.INFO)
CLASSICML_LOGGER = logging.getLogger(name='classicML')
CLASSICML_LOGGER.info('正在使用 {} 引擎'.format(os.environ['CLASSICML_ENGINE']))

from classicML.api import models
from classicML.api import plots

from classicML.api import BaseModel
from classicML.api import AveragedOneDependentEstimator
from classicML.api import AODE
from classicML.api import BackPropagationNeuralNetwork
from classicML.api import BPNN
from classicML.api import DecisionTreeClassifier
from classicML.api import LinearDiscriminantAnalysis
from classicML.api import LDA
from classicML.api import LogisticRegression
from classicML.api import NaiveBayesClassifier
from classicML.api import NB
from classicML.api import RadialBasisFunctionNetwork
from classicML.api import RBF
from classicML.api import SuperParentOneDependentEstimator
from classicML.api import SPODE
from classicML.api import SupportVectorClassifier
from classicML.api import SVC

from classicML.backend import activations
from classicML.backend import callbacks
from classicML.backend import data
from classicML.backend import initializers
from classicML.backend import io
from classicML.backend import kernels
from classicML.backend import losses
from classicML.backend import metrics
from classicML.backend import optimizers
from classicML.backend import tree

from classicML import benchmarks
