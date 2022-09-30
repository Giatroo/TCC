# Python Standard Libraries
import argparse
import math

# Third Party Libraries
import pandas as pd

# Project Libraries
from dataframes_loader import DataFramesLoader
from model_io import ModelIO
from predictions_evaluator import PredictionsEvaluator


def main(
    preloaded_data: bool = False,
    testing_instances: int = int(1e20),
    model_save_name: str = "deberta_base",
    verbose: bool = False,
):
    model_io = ModelIO(verbose=verbose)
    predictor = model_io.load_model(model_save_name)

    test_df = get_test_df(preloaded_data, testing_instances)
    X_test, y_test = get_X_y(test_df)
    y_hat = predictor.predict(X_test)  # type: ignore

    model_io.save_predictions(y_hat, model_save_name)  # type: ignore

    evaluator = PredictionsEvaluator(y_hat, y_test)  # type: ignore
    metrics = evaluator.general_evaluation()
    model_io.save_model_metrics(metrics, model_save_name)
    if verbose:
        print(metrics)


def transform_two_answers_per_row_to_one_answer_per_row(df):
    dfs = list()
    for i in range(1, 3):
        columns = ["comment_text", f"answer{i}_text", f"answer{i}_label"]
        map_columns = {
            f"answer{i}_text": "answer_text",
            f"answer{i}_label": "answer_label",
        }
        dfs.append(df[columns].rename(columns=map_columns))

    return pd.concat(dfs).reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    X = df[["comment_text", "answer_text"]].values
    X = list(map(tuple, X))

    y = df["answer_label"].values

    return X, y


def get_test_df(preloaded: bool, testing_instances: int) -> pd.DataFrame:
    df_loader = DataFramesLoader()
    _, test_df = df_loader.get_datasets(preloaded=preloaded)
    test_df = test_df[:testing_instances]
    test_df = transform_two_answers_per_row_to_one_answer_per_row(test_df)
    return test_df


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script trains a model on the train dataset and saves it."
    )
    parser.add_argument(
        "model_save_name",
        type=str,
        help="The name of the model to save. This is the only required argument.",
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
    )
    parser.add_argument(
        "--testing_instances",
        type=int,
        default=int(1e20),
        help="The number of instances to test on. If this number is above the number of rows in the testing dataset, the whole dataset will be used. The default is infinity.",
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


if __name__ == "__main__":
    args = parse_input()

    main(
        preloaded_data=args.preloaded_data,
        testing_instances=args.testing_instances,
        model_save_name=args.model_save_name,
        verbose=args.verbose,
    )
