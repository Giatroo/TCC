# Third Party Libraries
from sentence_transformers import CrossEncoder


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
    model = CrossEncoder(model_path)
    return model
