# Third Party Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Project Libraries
from dataframes_loader import DataFramesLoader
import model_io
from predictions_evaluator import PredictionsEvaluator

df_loader = DataFramesLoader()
train_df, test_df = df_loader.get_datasets()

texts = (
    train_df["comment_text"].values
    + train_df["answer1_text"].values
    + train_df["answer2_text"].values
)


def concat_commend_and_answer(
    df: pd.DataFrame, answer_number: int
) -> pd.Series:
    return df["comment_text"] + " | " + df[f"answer{answer_number}_text"]


train_texts1 = concat_commend_and_answer(train_df, 1)
train_texts2 = concat_commend_and_answer(train_df, 2)
train_texts = pd.concat([train_texts1, train_texts2], ignore_index=True)
train_labels = pd.concat(
    [train_df["answer1_label"], train_df["answer2_label"]], ignore_index=True
)


test_texts1 = concat_commend_and_answer(test_df, 1)
test_texts2 = concat_commend_and_answer(test_df, 2)
test_texts = pd.concat([test_texts1, test_texts2], ignore_index=True)
test_labels = pd.concat(
    [test_df["answer1_label"], test_df["answer2_label"]], ignore_index=True
)


def fit_predict(model, vectorizer):
    vectorizer = vectorizer.fit(texts)
    model.fit(vectorizer.transform(train_texts), train_labels)
    predictions = model.predict(vectorizer.transform(test_texts))

    evaluator = PredictionsEvaluator(predictions, test_labels)
    metrics = evaluator.general_evaluation()

    model_name = model.__class__.__name__
    vectorizer_name = vectorizer.__class__.__name__
    name = f"{model_name}_{vectorizer_name}"
    print(f"Saving metrics for {name}...")
    model_io.save_model_metrics(metrics, name)


vectorizer = CountVectorizer(stop_words="english")
nb = MultinomialNB()
fit_predict(nb, vectorizer)
