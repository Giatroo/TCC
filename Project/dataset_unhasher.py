import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
from pandas import DataFrame, Series

import utils


def read_balanced_data(train: bool = True) -> DataFrame:
    """Reads the balanced dataset from the dataset path and sets its column names.

    Parameters
    ----------
    train : bool, default=True
        If it's train or test.

    Raises
    ------
    FileNotFoundError
        if the dataset doesn't exist in the expected path (indicated in the
        global variables json file).


    Returns
    -------
    DataFrame
        The read dataframe.
    """
    dataset_path = utils.get_global_vars()["dataset_path"]
    train_test_str = "train" if train else "test"
    file_path = f"{dataset_path}/{train_test_str}-balanced.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = pd.read_csv(file_path, sep="|", header=None)
    df = df.rename(columns={0: "discution", 1: "targets", 2: "labels"})
    return df


def filter_rows_with_more_than_one_post(df: DataFrame) -> DataFrame:
    """Filters the dataframe discution column to keep only rows with one post.

    Parameters
    ----------
    df : DataFrame
        The dataframe to filter.

    Returns
    -------
    DataFrame
        The filtered dataframe.
    """
    filtered_df = df[df["discution"].str.split(" ").apply(len) == 1]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def load_comments_json():
    """Loads the JSON file that maps each hashed value into a dictionary with
    attributes 'text', 'author', 'score', 'ups', 'downs', 'date',
    'created_utc',  and 'subreddit'.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary with the hashed values as keys and a dictionary with the
        comment attributes as values.
    """
    dataset_path = utils.get_global_vars()["dataset_path"]
    comments = json.load(open(f"{dataset_path}/comments.json"))
    return comments


def get_n_instances_from_df(df: DataFrame, num_instances: int) -> DataFrame:
    """Receives a dataframe and a number of instances to return.

    Returns the first num_instances rows of the dataframe (different strategies
    could be used instead).

    Parameters
    ----------
    df : DataFrame
        Any dataframe
    num_instances : int
        The number of lines to return.

    Returns
    -------
    DataFrame
        A dataframe with only num_instances rows.
    """
    if num_instances > len(df):
        return df
    sub_df = df.iloc[:num_instances]
    return sub_df


def save_unhashed_df_to_path(df: DataFrame, path: str):
    """Receives a dataframe and a path to save it.

    Parameters
    ----------
    df : DataFrame
        A dataframe to save.
    path : str
        The path to save (without the file extension).
    """
    df.to_pickle(f"{path}.pkl")


def translate_row(
    row: Series,
    comments: Dict[str, Dict[str, Any]],
    keys_of_interest: List[str],
) -> Series:
    """Receives a row Series with keys 'discution', 'targets', and 'labels', a
    dictionary that maps each hashed value into the comment attributes, and a
    list of keys of interest to return from the comment dictionary.

    The function returns a new row with the discution and targets attributes
    instead of the hashed values. Only the attributes in the keys of interest
    list are kept.

    Parameters
    ----------
    row : Series
        A series with keys 'discution', 'targets', and 'labels'. The value of
        'discution' is a single hash value, the value of 'targets' is a string
        with two hash values, and the value of 'labels' is either the string
        '1 0' or '0 1'.
    comments : Dict[str, Dict[str, Any]]
        A dictionary in which the keys are the hash values present in the row
        and the value is another dictionary containing the comment attributes.
        The attributes are 'text', 'author', 'score', 'ups', 'downs', 'date',
        'created_utc', and 'subreddit'.
    keys_of_interest : List[str]
        A list of keys retrieve for each comment (the discution and the
        targets).

    Returns
    -------
    new_row : Series
        Returns a new row with the keys 'comment_*', 'target1_*', 'target2_*',
        'target1_label', and 'target2_label', where the '*' means all the keys
        present in the keys_of_interest list. Also, if 'subreddit' is in
        keys_of_interest, it returns only a 'subreddit' key instead of
        'comment_subreddit', 'target1_subreddit', and 'target2_subreddit'
        (since the subreddit is always the same).
    """
    # Defining if we'll get the subreddit key
    get_subreddit = "subreddit" in keys_of_interest
    if get_subreddit:
        keys_of_interest.remove("subreddit")

    # Getting the info for the discution comment
    discution_hash = row["discution"]
    discution_info = utils.filter_dict_by_keys(
        comments[discution_hash], keys_of_interest
    )
    new_row_dict = {
        f"comment_{key}": value for key, value in discution_info.items()
    }

    # Getting the info for the targets comments
    target_hashs = row["targets"].split()
    for i, target_hash in enumerate(target_hashs):
        target_info = utils.filter_dict_by_keys(
            comments[target_hash], keys_of_interest
        )
        for key, value in target_info.items():
            new_row_dict[f"answer{i+1}_{key}"] = value

    # Getting the subreddit info
    if get_subreddit:
        new_row_dict["subreddit"] = comments[discution_hash]["subreddit"]

    # Getting the labels info
    labels = row["labels"].split()
    new_row_dict["answer1_label"] = int(labels[0])
    new_row_dict["answer2_label"] = int(labels[1])

    # Defining the new row as a Series
    new_row = pd.Series(new_row_dict)
    return new_row


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script parses the dataset and saves it in a pickle file."
    )
    parser.add_argument(
        "--num_instances",
        "-n",
        type=int,
        required=True,
        help="The number of instances to preprocess.",
    )
    args = parser.parse_args()
    return args


def main(num_instances: int):
    """The main function.

    It's called when we call the module as a script.

    It must receive an integer as argument, which is the number of instances to
    unhash from the dataset. Also the dataset path must be already downloaded
    in the right folder (which is the folder defined in the global variables).

    Parameters
    ----------
    num_instances : int
        The number of instances to preprocess.
    """

    dataset_path = utils.get_global_vars()["dataset_path"]
    keys_of_interest = [
        "text",
    ]

    print("Loading comments json... This might take a while...")
    comments = load_comments_json()

    for train_bool, train_str in zip([True, False], ["train", "test"]):
        print("-" * 40 + "\n")

        print(f"Reading balanced {train_str} data...")
        df = read_balanced_data(train=train_bool)
        df = filter_rows_with_more_than_one_post(df)

        print(f"Unhashing first {num_instances} rows...")
        sub_df = get_n_instances_from_df(df, num_instances)
        sub_df = sub_df.apply(
            translate_row, axis=1, args=(comments, keys_of_interest)
        )

        file_path = f"{dataset_path}balanced_unhashed_{train_str}_df"
        print(f"Saving to '{file_path}'...")
        save_unhashed_df_to_path(sub_df, file_path)


if __name__ == "__main__":
    args = parse_input()
    num_instances = args.num_instances

    main(num_instances)
