"""An easy-to-use ML framework"""
__version__ = '0.2.3'

from classicML.DecisionTree import *
from classicML.NeuralNetwork import *

from classicML.DecisionTree.tree_model.decision_tree import DecisionTree

from classicML.NeuralNetwork.nn_model.back_propagation_neural_network import BackPropagationNeuralNetwork
from classicML.NeuralNetwork.nn_model.back_propagation_neural_network import BackPropagationNeuralNetwork as BPNN
from classicML.NeuralNetwork.nn_model.radial_basis_function_network import RadialBasisFuncionNetwork
from classicML.NeuralNetwork.nn_model.radial_basis_function_network import RadialBasisFuncionNetwork as RBF

# plot
from classicML.DecisionTree.tree_plot.plot_tree import plot_decision_tree
from classicML.NeuralNetwork.nn_plot.plot_nn import plot_history