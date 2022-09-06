"""This script is used to get a valid model after the training. Many times, we
train a model and it is not trained well. The model predicts the same labels for
all the testing instances. This script will train a model on the training data,
and make sure that it's accuracy is above a certain threshold.

It will generate a log of the training process with all the models and their
accuracies.

At the end, it stop when the model is trained well and save it.
"""

# Python Standard Libraries
import argparse
import os

# Project Libraries
import model_io
import run_tester
import run_trainer
import utils

THRESHOLD = 0.55


def _parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script tests a model on the test dataset. It receives a pretrained model and uses it to predict and test the predictions."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The model name. This is the only required argument.",
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
    )
    parser.add_argument(
        "--bert",
        action="store_true",
        default=False,
        help="Whether to use BERT or DeBERTa. If not specified, DeBERTa is used.",
    )
    parser.add_argument(
        "--warmup_steps",
        "-ws",
        type=int,
        default=100,
        help="The number of warmup steps for the model. The default is 100.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="The batch size for the dataloader. The default is 8.",
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


def main(model_name: str, **kwargs) -> None:
    preloaded_data = kwargs.get("preloaded_data", False)
    use_bert = kwargs.get("bert", False)
    warmup_steps = kwargs.get("warmup_steps", 100)
    batch_size = kwargs.get("batch_size", 8)
    verbose = kwargs.get("verbose", False)

    while True:
        run_trainer.main(
            model_name,
            preloaded_model=False,
            epochs=1,
            preloaded_data=preloaded_data,
            use_bert=use_bert,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            verbose=verbose,
        )
        metrics = run_tester.get_metrics(
            model_name, preloaded_data=preloaded_data, verbose=verbose
        )
        acc = metrics["accuracy"]

        print("=" * 40)
        print(f"Accuracy: {acc}")
        print("=" * 40)

        model_io.save_model_metrics(metrics, model_name)

        if acc > THRESHOLD:
            break

        models_path = utils.get_global_vars()["models_path"]
        os.system(f"rm -rf {models_path}/{model_name}")


if __name__ == "__main__":
    args = _parse_input()

    main(
        model_name=args.model_name,
        preloaded_data=args.preloaded_data,
        use_bert=args.bert,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
