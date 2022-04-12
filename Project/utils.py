"""This module has global utilities for the project."""
import json
from typing import Any, List

DEFAULT_GLOBAL_VARS_PATH = "global_vars.json"


def get_global_vars(
    json_path: str = DEFAULT_GLOBAL_VARS_PATH,
) -> dict[str, str]:
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


def get_keys_of_interest_from_dict(
    dictionary: dict[Any, Any], keys_of_interest: List[Any]
) -> dict[Any, Any]:
    """Receives a dictionary and a list of keys and returns a dictionary with only the keys of interest.

    Parameters
    ----------
    dictionary : dict[Any, Any]
        Any dictionary.
    keys_of_interest : List[Any]
        A list of keys to keep.

    Returns
    -------
    dict[Any, Any]
        The original dictionary, but only with the keys present in the list.
    """
    filtered_dict = {
        key: dictionary[key] for key in keys_of_interest if key in dictionary
    }
    return filtered_dict
