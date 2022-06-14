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

        predictions, y_true = predictor.get_probabilities_and_labels(
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


def parse_input():
    # Python Standard Libraries
    import argparse

    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script tests a model on the test dataset. It receives a pretrained model and uses it to predict and test the predictions."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of a pretrained model.",
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Whether to print the progress or not.",
    )

    args = parser.parse_args()
    return args


def main(
    model_name: str,
    preloaded_data: bool,
    verbose: bool,
) -> None:
    """Trains a model and saves it.

    Parameters
    ----------
    model_name : str
        The name of the model to save.
    preloaded_data : bool
        Whether to use the preloaded data or not.
    verbose : bool
        Whether to print the progress or not.
    """
    # Project Libraries
    from dataframes_loader import DataFramesLoader
    from predictions_evaluator import PredictionsEvaluator

    df_loader = DataFramesLoader()
    _, test_df = df_loader.get_datasets(preloaded_data)
    models_path = utils.get_global_vars()["models_path"]
    evaluations_path = utils.get_global_vars()["evaluations_path"]

    model = CrossEncoder(f"{models_path}{model_name}")
    tester = ModelTester(model, verbose)
    metrics = tester.get_model_metrics(test_df)

    if verbose:
        print(
            f"Saving evaluation metrics to {evaluations_path}{model_name}.txt"
        )
    tester.save_model_metrics(metrics, model_name)


if __name__ == "__main__":
    args = parse_input()

    main(
        model_name=args.model_name,
        preloaded_data=args.preloaded_data,
        verbose=args.verbose,
    )
