# Python Standard Libraries
import math

# Third Party Libraries
import ktrain
from ktrain import text
import pandas as pd
from sklearn.model_selection import train_test_split

# Project Libraries
from dataframes_loader import DataFramesLoader
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


def get_train_df(train_rows_cap: int) -> pd.DataFrame:
    df_loader = DataFramesLoader()
    train_df, _ = df_loader.get_datasets(preloaded=True)
    train_df = train_df[:train_rows_cap]
    train_df = transform_two_answers_per_row_to_one_answer_per_row(train_df)
    return train_df


def get_train_validation_df(
    train_df: pd.DataFrame,
    validation_perc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size_split = int(len(train_df) * validation_perc)
    train_df, validation_df = train_test_split(
        train_df, test_size=size_split, random_state=42  # type: ignore
    )
    train_df.reset_index(inplace=True, drop=True)  # type: ignore
    validation_df.reset_index(inplace=True, drop=True)  # type: ignore
    return train_df, validation_df  # type: ignore


def main(
    train_rows_cap: int = int(math.inf),
    validation_perc: float = 0.15,
    model_name="microsoft/deberta-base",
    model_save_name="deberta_base",
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float = 5e-5,
):
    print(train_rows_cap)
    train_df = get_train_df(train_rows_cap)
    train_df, validation_df = get_train_validation_df(train_df, validation_perc)

    t = text.Transformer(
        model_name,
        maxlen=512,
        class_names=train_df["answer_label"].unique().tolist(),
    )

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(validation_df)

    trn = t.preprocess_train(X_train, y_train)
    val = t.preprocess_train(X_val, y_val)

    model = t.get_classifier()
    learner = ktrain.get_learner(
        model, train_data=trn, val_data=val, batch_size=batch_size
    )

    learner.autofit(learning_rate, epochs)

    models_path = utils.get_global_vars()["models_path"]
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    predictor.save(f"{models_path}{model_save_name}")
