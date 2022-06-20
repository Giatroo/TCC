# Python Standard Libraries
from datetime import datetime
from typing import Dict

# Third Party Libraries
from pandas import DataFrame
from sentence_transformers import CrossEncoder

# Project Libraries
from model_predictor import Predictor
from predictions_evaluator import PredictionsEvaluator
import utils


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

    def save_model_metrics(self, metrics: Dict[str, float], name: str) -> None:
        evaluations_path = utils.get_global_vars()["evaluations_path"]
        with open(f"{evaluations_path}/{name}.txt", "a") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== {now} ===\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
