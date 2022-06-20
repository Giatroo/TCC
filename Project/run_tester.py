# Python Standard Libraries
import argparse

# Project Libraries
from dataframes_loader import DataFramesLoader
import model_io
from model_tester import ModelTester
from predictions_evaluator import PredictionsEvaluator
import utils


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script tests a model on the test dataset. It receives a pretrained model and uses it to predict and test the predictions."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of a pretrained model. This is the only required argument.",
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
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


def main(
    model_name: str,
    preloaded_data: bool,
    verbose: bool,
) -> None:
    """Trains a model and saves it.

    Parameters
    ----------
    model_name : str
        The name of the model to save.
    preloaded_data : bool
        Whether to use the preloaded data or not.
    verbose : bool
        Whether to print the progress or not.
    """

    df_loader = DataFramesLoader()
    _, test_df = df_loader.get_datasets(preloaded_data)
    models_path = utils.get_global_vars()["models_path"]
    model_path = f"{models_path}{model_name}"
    evaluations_path = utils.get_global_vars()["evaluations_path"]

    model = model_io.load_model(model_path)
    tester = ModelTester(model, verbose)
    metrics = tester.get_model_metrics(test_df)

    if verbose:
        print(
            f"Saving evaluation metrics to {evaluations_path}{model_name}.txt"
        )
    tester.save_model_metrics(metrics, model_name)


if __name__ == "__main__":
    args = parse_input()

    main(
        model_name=args.model_name,
        preloaded_data=args.preloaded_data,
        verbose=args.verbose,
    )
