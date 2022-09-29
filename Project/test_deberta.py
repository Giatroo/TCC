# Python Standard Libraries
import math
import pickle

# Third Party Libraries
import ktrain
from ktrain import text
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project Libraries
from dataframes_loader import DataFramesLoader
import model_io as io
from predictions_evaluator import PredictionsEvaluator
import utils


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


def get_test_df(test_rows_cap: int) -> pd.DataFrame:
    df_loader = DataFramesLoader()
    _, test_df = df_loader.get_datasets(preloaded=True)
    test_df = test_df[:test_rows_cap]
    test_df = transform_two_answers_per_row_to_one_answer_per_row(test_df)
    return test_df


def main(test_rows_cap: int = int(math.inf), model_save_name="deberta_base"):
    test_df = get_test_df(test_rows_cap)

    X_test, y_test = get_X_y(test_df)
    models_path = utils.get_global_vars()["models_path"]
    predictor = ktrain.load_predictor(f"{models_path}{model_save_name}")

    y_hat = predictor.predict(X_test)  # type: ignore

    predictions_path = utils.get_global_vars()["predictions_path"]
    with open(f"{predictions_path}{model_save_name}.pkl", "wb") as f:
        pickle.dump(y_hat, f)

    evaluator = PredictionsEvaluator(y_hat, y_test)
    metrics = evaluator.general_evaluation()
    io.save_model_metrics(metrics, model_save_name)
    print(metrics)
