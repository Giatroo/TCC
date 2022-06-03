# Python Standard Libraries
import typing
from typing import Dict, Iterable

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
    _predictions: Iterable
    _true_values: Iterable

    def __init__(self, predictions: Iterable, true_values: Iterable) -> None:
        self._predictions = predictions
        self._true_values = true_values

    MetricFunc = typing.Callable[[Iterable, Iterable], float]

    def evaluate_using(self, metric_func: MetricFunc) -> float:
        return metric_func(self._true_values, self._predictions)

    def general_evaluation(self) -> Dict[str, float]:
        return {
            "accuracy": self.evaluate_using(accuracy_score),
            "precision": self.evaluate_using(precision_score),
            "recall": self.evaluate_using(recall_score),
            "f1": self.evaluate_using(f1_score),
            "roc_auc": self.evaluate_using(roc_auc_score),
            "confusion_matrix": self.evaluate_using(confusion_matrix),
        }
