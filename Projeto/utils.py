"""This module has global utilities for the project."""
import json

DEFAULT_GLOBAL_VARS_PATH = 'global_vars.json'

def get_global_vars(json_path: str = DEFAULT_GLOBAL_VARS_PATH) -> dict[str, str]:
    """Returns the global variables from the json file.

    Parameters
    ----------
    json_path : str, optional
        The path to the json file containing the global variables, by default DEFAULT_GLOBAL_VARS_PATH

    Returns
    -------
    dict
        The dictionary containing the global variables.
    """
    return json.load(open(json_path))