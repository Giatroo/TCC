# Python Standard Libraries
from datetime import datetime
import pickle
from typing import Dict, Iterable

# Third Party Libraries
import ktrain
from ktrain.predictor import Predictor

# Project Libraries
import utils


class ModelIO:
    _verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def save_model(self, model: Predictor, model_save_name: str) -> None:
        """Saves a model.

        Parameters
        ----------
        model : Predictor
            The model to save.
        model_path : str
            The path to save the model. It must be a folder name.
        """
        models_path = utils.get_global_vars()["models_path"]
        save_path = f"{models_path}{model_save_name}"

        if self._verbose:
            print(f"Saving model to {save_path}...")

        model.save(save_path)

    def load_model(self, model_save_name: str) -> Predictor:
        """Loads a model.

        Parameters
        ----------
        model_save_name : str
            The name of the model when it was saved.

        Returns
        -------
        model : CrossEncoder
            The pre-trained model.
        """
        models_path = utils.get_global_vars()["models_path"]
        load_path = f"{models_path}{model_save_name}"

        if self._verbose:
            print(f"Loading model from {load_path}...")

        predictor = ktrain.load_predictor(load_path)
        return predictor

    def save_predictions(self, y_hat: Iterable, model_save_name: str) -> None:
        predictions_path = utils.get_global_vars()["predictions_path"]
        with open(f"{predictions_path}{model_save_name}.pkl", "wb") as f:
            pickle.dump(y_hat, f)

    def save_model_metrics(
        self, metrics: Dict[str, float], model_save_name: str
    ) -> None:
        evaluations_path = utils.get_global_vars()["evaluations_path"]
        evaluation_path = f"{evaluations_path}/{model_save_name}.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self._verbose:
            print(f"Saving evaluations to {evaluation_path}...")

        with open(evaluation_path, "a") as f:
            f.write(f"=== {now} ===\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
