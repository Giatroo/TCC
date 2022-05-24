"""Module to download the preloaded datasets from Google Drive.

The preloaded datasets are training and testing datasets that are already preprocessed and ready to be used to train and test the models."""

import os

import gdown

import utils


def download_file_from_drive(file_name, file_url):
    """Downloads a file from Google Drive.

    Parameters
    ----------
    file_name : str
        The name of the file to download.
    file_url : str
        The URL of the file to download.
    """
    dataset_path = utils.get_global_vars()["dataset_path"]
    gdown.download(file_url, f"{dataset_path}{file_name}", quiet=False)


def download_preloaded_datasets():
    """Downloads the preloaded datasets from Google Drive."""

    preloaded_files_info = utils.get_global_vars()["preloaded_files_info"]
    for preloaded_file_info in preloaded_files_info:
        file_name = preloaded_file_info["file_name"]
        file_url = preloaded_file_info["file_url"]
        download_file_from_drive(file_name, file_url)


def main():
    dataset_path = utils.get_global_vars()["dataset_path"]
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    download_preloaded_datasets()


if __name__ == "__main__":
    main()
