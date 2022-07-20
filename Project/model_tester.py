# Python Standard Libraries
from typing import Dict

# Third Party Libraries
from pandas import DataFrame
from sentence_transformers import CrossEncoder

# Project Libraries
from model_predictor import Predictor
from predictions_evaluator import PredictionsEvaluator


class ModelTester:
    _model: CrossEncoder
    _verbose: bool

    def __init__(self, model: CrossEncoder, verbose: bool = False) -> None:
        self._model = model
        self._verbose = verbose

    def get_model_metrics(self, dataset: DataFrame) -> Dict[str, float]:
        predictor = Predictor()

        predictions, y_true = predictor.get_probabilities_and_true_labels(
            dataset,
            self._model,
            verbose=self._verbose,
        )
        y_pred = predictor.get_labels_from_probs(predictions)

        evaluator = PredictionsEvaluator(y_pred, y_true)
        return evaluator.general_evaluation()
