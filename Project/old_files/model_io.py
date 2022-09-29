# Python Standard Libraries
from datetime import datetime
from typing import Dict

# Third Party Libraries
from sentence_transformers import CrossEncoder

# Project Libraries
import utils


def save_model(model: CrossEncoder, model_path: str) -> None:
    """Saves a model.

    Parameters
    ----------
    model : CrossEncoder
        The model to save.
    model_path : str
        The path to save the model. It must be a folder name.
    """
    model.save(model_path)


def load_model(model_path: str) -> CrossEncoder:
    """Loads a model.

    Parameters
    ----------
    model_path : str
        The path to the model. It must be a folder name.

    Returns
    -------
    model : CrossEncoder
        The pre-trained model.
    """
    model = CrossEncoder(model_path, num_labels=2)
    return model


def save_model_metrics(metrics: Dict[str, float], name: str) -> None:
    evaluations_path = utils.get_global_vars()["evaluations_path"]
    with open(f"{evaluations_path}/{name}.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"=== {now} ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
