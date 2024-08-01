import numpy as np

from abc import ABC, abstractmethod


class ClassifierStatsMixin(ABC):
    def __init__(self, estimator, X, y):
        self.X, self.y = np.array(X), np.array(y)
        self.intercept = estimator.intercept_

        self.y_pred = estimator.predict(self.X)


    @abstractmethod
    def accuracy(self):
        """Accuracy of the model."""


    @abstractmethod
    def recall(self):
        """Precision of the model."""


    @abstractmethod
    def precision(self):
        """Precision of the model."""


    @abstractmethod
    def f1_score(self):
        """Harmonic mean of the model."""


    @staticmethod
    def roc_auc(self):
        pass


class BinaryClassifierStatistics(ClassifierStatsMixin):
    def accuracy(self):
        """Accuracy of the model."""

        num_of_correct = np.sum(self.y == self.y_pred)
        accuracy = np.mean(num_of_correct)

        return accuracy


    def recall(self):
        pass


    def precision(self):
        pass


    def f1_score(self):
        pass








