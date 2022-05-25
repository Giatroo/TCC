# Python Standard Libraries
import typing

# Third Party Libraries
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class PredictionsEvaluator:
    _predictions: ndarray
    _true_values: ndarray

    def __init__(self, predictions: ndarray, true_values: ndarray):
        self._predictions = predictions
        self._true_values = true_values

    MetricFunc = typing.Callable[[ndarray, ndarray], float]

    def evaluate_using(self, metric_func: MetricFunc) -> float:
        return metric_func(self._true_values, self._predictions)

    def general_evaluation(self):
        return {
            "accuracy": self.evaluate_using(accuracy_score),
            "precision": self.evaluate_using(precision_score),
            "recall": self.evaluate_using(recall_score),
            "f1": self.evaluate_using(f1_score),
            "roc_auc": self.evaluate_using(roc_auc_score),
            "confusion_matrix": self.evaluate_using(confusion_matrix),
        }
