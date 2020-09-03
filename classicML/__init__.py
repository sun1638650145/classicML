"""An easy-to-use ML framework"""
__version__ = '0.4b1'

from .DecisionTree import *
from .NeuralNetwork import *
from .SupportVectorMachine import *

from .DecisionTree.tree_model.decision_tree import DecisionTree

from .NeuralNetwork.nn_model.back_propagation_neural_network import BackPropagationNeuralNetwork
from .NeuralNetwork.nn_model.back_propagation_neural_network import BackPropagationNeuralNetwork as BPNN
from .NeuralNetwork.nn_model import optimizers
from .NeuralNetwork.nn_model import losses
from .NeuralNetwork.nn_model.radial_basis_function_network import RadialBasisFuncionNetwork
from .NeuralNetwork.nn_model.radial_basis_function_network import RadialBasisFuncionNetwork as RBF

from .SupportVectorMachine.svm_model.support_vector_classification import SupportVectorClassification
from .SupportVectorMachine.svm_model.support_vector_classification import SVC

from .LinearModel.lm_model.logistic_regression import LogisticRegression

# plot
from .DecisionTree.tree_plot.plot_tree import plot_decision_tree
from .NeuralNetwork.nn_plot.plot_nn import plot_history
from .SupportVectorMachine.svm_plot.plot_svm import plot_svc
from .LinearModel.lm_plot.plot_lr import plot_logistic_regression