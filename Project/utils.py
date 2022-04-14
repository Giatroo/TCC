"""This module has global utilities for the project."""
import json
import typing
from typing import Any, Dict, List

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


KeyType = typing.TypeVar("KeyType")
ValueType = typing.TypeVar("ValueType")


def filter_dict_by_keys(
    dictionary: Dict[KeyType, ValueType], filter_keys: List[KeyType]
) -> Dict[KeyType, ValueType]:
    """Receives a dictionary and a list of keys and returns a dictionary with only the keys of interest.

    Parameters
    ----------
    dictionary : dict[KeyType, ValueType]
        Any dictionary.
    filter_keys : List[KeyType]
        A list of keys to keep.

    Returns
    -------
    Dict[KeyType, ValueType]
        The original dictionary, but only with the keys present in the list.
    """
    filtered_dict = {
        key: dictionary[key] for key in filter_keys if key in dictionary
    }
    return filtered_dict
