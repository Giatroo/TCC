import argparse

import dataset_downloader as downloader
import dataset_unhasher as unhasher


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Downloads and preprocesses the dataset.",
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
    """

    downloader.main()
    unhasher.main(num_instances)


if __name__ == "__main__":
    args = parse_input()
    num_instances = args.num_instances
    main(num_instances)
