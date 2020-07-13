import numpy as np


class History:
    """训练记录"""
    def __init__(self, epochs):
        self.loss = np.ones(epochs) * float('inf')
        self.acc = np.zeros(epochs)

    def add_loss(self, loss):
        self.loss = np.asarray(loss)

    def add_accuracy(self, acc):
        self.acc = np.asarray(acc)
