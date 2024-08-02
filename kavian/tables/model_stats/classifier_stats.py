import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix

def _divide(numerator, denominator):
    """
    Performs division & handles Zero Division errors by replacing
    errors with zero, a common convention with classifier metrics.
    """

    if denominator == 0:
        return 0 # Instead of ZeroDivision Error

    division = np.divide(numerator, denominator)

    return division


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
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        CM = confusion_matrix(self.y, self.y_pred)

        # Flatten to a 1D-array and unpack
        self.true_neg, self.false_pos, \
        self.false_neg, self.true_pos = CM.ravel()

        self.recall = self.recall()
        self.precision = self.precision()


    def accuracy(self):
        """
        Calculates the accuracy of the classifier model. Accuracy finds the proportion
        of correctly specified labels (true positives & true negatives) out of all
        predictions for a response vector y.
        """

        num_of_correct = self.true_pos + self.true_neg
        accuracy = num_of_correct / len(self.y_pred)

        return accuracy


    def recall(self):
        """
        Calculates the recall of the classifier model, or the ability for the model
        to correctly identify all positive labels in the dataset.

        Recall finds the proportion of correctly specified positives out of the
        entire positive sample space of a response vector y.
        """

        total_pos = self.true_pos + self.false_neg

        # Handle zero division
        recall = _divide(self.true_pos, total_pos)

        return recall


    def precision(self):
        """
        Caculates the precision of the classifier model, or the proportion of
        predicted positives that were actually positive.

        To do so, simply express the number of true positives as a proportion
        of the sample space of our model's positive predictions.
        """

        total_pos_predictions = self.true_pos + self.false_pos

        # Handle zero division
        precision = _divide(self.true_pos, total_pos_predictions)

        return precision


    def f1_score(self):
        """
        Calculates harmonic mean of precision & recall.

        Because precision and recall are bounded by a trade-off, the
        F1 Score strikes a balance between the two measures. This metric
        is particularly useful in the presence of an unbalanced dataset.
        """

        # Handle zero division
        f1_score = _divide(2 * self.precision * self.recall, self.precision + self.recall)

        return f1_score


    def mcc(self):
        """
        Calculates Matthew's Correlation Coefficient.

        MCC is particularly helpful for binary classification. It takes
        all outputs from the confusion matrix into account, resulting in
        a balanced measure that can be used even when the classes are of
        starkly different sizes. Matthew's Correlation Coefficient mirrors
        that of the Pearson Correlation Coefficient, such that:

        - A value near 1 corresponds to near-perfect predictions
        - A value near 0 corresponds to a model no better than
          random guessing
        - A value near -1 corresponds to a model whose predictions
          are always incorrect.
        """

        tp, fp = self.true_pos, self.false_pos
        tn, fn = self.true_neg, self.false_neg

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Handle zero division
        mcc = _divide(numerator, denominator)

        return mcc












