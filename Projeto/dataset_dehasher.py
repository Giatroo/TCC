import json
import sys
from ctypes import util
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

import utils


def read_balanced_data(train: bool = True) -> DataFrame:
    """Reads the balanced dataset from the dataset path and sets its column names.

    Parameters
    ----------
    train : bool, optional
        If it's train or test, by default True

    Returns
    -------
    DataFrame
        The read dataframe.
    """
    GLOBAL_VARS = utils.get_global_vars()
    train_test_str = "train" if train else "test"
    df = pd.read_csv(
        f"{GLOBAL_VARS['dataset_path']}/{train_test_str}-balanced.csv",
        sep="|",
        header=None,
    )
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
    return df[df["discution"].str.split(" ").apply(len) <= 1].reset_index(
        drop=True
    )


def load_comments_json():
    GLOBAL_VARS = utils.get_global_vars()
    comments = json.load(open(f'{GLOBAL_VARS["dataset_path"]}/comments.json'))
    return comments


def get_top_comment_info(
    top_comment: str, comments: dict[str, Any]
) -> dict[str, Any]:
    keys_of_interest = [
        "text",
        "author",
        #'subreddit',
        "score",
        #'ups',
        #'downs',
        #'date',
        #'created_utc'
    ]
    top_comment_info = utils.get_keys_of_interest_from_dict(
        comments[top_comment], keys_of_interest
    )
    return top_comment_info


def get_target_comment_info(
    target_comment: str, comments: dict[str, Any]
) -> dict[str, Any]:
    keys_of_interest = [
        "text",
        "author",
        #'subreddit',
        "score",
        #'ups',
        #'downs',
        #'date',
        #'created_utc'
    ]
    target_comment_info = utils.get_keys_of_interest_from_dict(
        comments[target_comment], keys_of_interest
    )
    return target_comment_info


def translate_line(row: Series, comments: dict[str, Any]) -> Series:
    top_comment = row[0]
    target_comment1 = row[1].split()[0]
    target_comment2 = row[1].split()[1]
    top_comment_info = get_top_comment_info(top_comment, comments)
    target_comment1_info = get_target_comment_info(target_comment1, comments)
    target_comment2_info = get_target_comment_info(target_comment2, comments)
    subreddit = comments[top_comment]["subreddit"]

    line_dict = dict()
    info_list = [top_comment_info, target_comment1_info, target_comment2_info]
    name_list = ["comment", "answer1", "answer2"]

    for info_dict, name in zip(info_list, name_list):
        for key, info in info_dict.items():
            line_dict[f"{name}_{key}"] = info
    line_dict["subreddit"] = subreddit
    line_series = pd.Series(line_dict)
    return line_series


def main():
    GLOBAL_VARS = utils.get_global_vars()

    if len(sys.argv) == 1:
        print("Usage: python dataset_dehasher.py <num_instances>")
        exit(-1)
    try:
        num_instances = int(sys.argv[1])
    except ValueError:
        print("Please provide an integer number of instances.")
        exit()

    print("Loading comments json...")
    comments = load_comments_json()

    for train_bool, train_str in zip([True, False], ["train", "test"]):
        print("-" * 40 + "\n")
        print(f"Reading balanced {train_str} data...")
        df = read_balanced_data(train=train_bool)
        df = filter_rows_with_more_than_one_post(df)
        print(f"Dehashing first {num_instances} rows...")
        df = df.iloc[:num_instances, :].apply(
            translate_line, axis=1, args=(comments,)
        )
        print(df)
        print(f"Saving to '{train_str}'...")
        df.to_csv(
            f"{GLOBAL_VARS['dataset_path']}/balanced_{train_str}_df.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
