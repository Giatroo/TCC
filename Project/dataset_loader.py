"""This module has functionally to get the dataset from the web and unpack it.
    One can run directly it to get the dataset from the web and unpack it.
    
    It uses dataset_url and dataset_path from the global variables file to
    know the URL to extract the data and the path to put it.
"""
import bz2
import os

import utils


def move_dataset_to_right_path(num_cut_dirs: int):
    """Moves the dataset to the path specified by the global variables.

    Parameters
    ----------
    num_cut_dirs : int
        The number of directiories to cut from the path.
    """
    GLOBAL_VARS = utils.get_global_vars()

    nested_folders = GLOBAL_VARS["dataset_url"].split("/")[3:]
    nested_folders = nested_folders[num_cut_dirs:]
    cur_dataset_path = "/".join(nested_folders)

    print(f'mv {cur_dataset_path} {GLOBAL_VARS["dataset_path"]}')
    os.system(f'mv {cur_dataset_path} {GLOBAL_VARS["dataset_path"]}')

    if os.path.exists(cur_dataset_path.split("/")[0]):
        print(f'rm -rf {cur_dataset_path.split("/")[0]}')
        os.system(f'rm -rf {cur_dataset_path.split("/")[0]}')


def get_dataset_from_web(url: str, num_cut_dirs: int):
    """Downloads the dataset from the url and puts it in the path specified by
    the global variable dataset_path.

    Parameters
    ----------
    url : str
        The url to download the dataset from.
    num_cut_dirs : int
        The number of directories to cut from the url path.
    """
    os.system(f"wget -r -np -nH --cut-dirs={num_cut_dirs} -R index.html* {url}")
    move_dataset_to_right_path(num_cut_dirs)


def unpack_bz2_files(path: str):
    """Unpacks all the bz2 files recursively in the path.

    Parameters
    ----------
    path : str
        The path to unpack the bz2 files.
    """
    for file in os.listdir(path):
        if os.path.isdir(path + file):
            unpack_bz2_files(path + file + "/")
        else:
            if not file.endswith(".bz2"):
                continue
            print(f"Unpacking {path + file}")
            with bz2.open(path + file, "rb") as f:
                with open(path + file[:-4], "wb") as out_file:
                    out_file.write(f.read())


def delete_bz2_files(path: str):
    """Deletes all the bz2 files recursively in the path.

    Parameters
    ----------
    path : str
        The path to delete the bz2 files.
    """
    for file in os.listdir(path):
        if os.path.isdir(path + file):
            delete_bz2_files(path + file + "/")
        else:
            if not file.endswith(".bz2"):
                continue
            print(f"Deleting {path + file}")
            os.remove(path + file)


def check_if_path_already_exists(path: str) -> bool:
    """Check if a dataset path already exists. If it does, it asks the user
    if he wants to delete it and re-download it.

    Parameters
    ----------
    path : str
        The path to check if it already exists.

    Returns
    -------
    bool
        If the path already exists.
    """
    if not os.path.exists(path):
        return False

    print(f"The dataset path {path} already exists.")
    k = input("Do you want to delete it and re-download the dataset? (y/[n]) ")
    if k != "y":
        return True

    print("Deleting the dataset...")
    os.system(f"rm -rf {path}")
    return False


def main():
    """The main function of the module.
    It gets and dataset from the web, puts it in the path specified by the global
    variables, unpacks the bz2 files, and deletes them.
    """
    # importing inside code because we don't use it in the rest of the module
    import sys

    if len(sys.argv) > 1:
        try:
            num_cut_dirs = int(sys.argv[1])
        except ValueError:
            print("Please provide an integer number of cut directories.")
            exit()
    else:
        num_cut_dirs = 0

    GLOBAL_VARS = utils.get_global_vars()

    path_already_exists = check_if_path_already_exists(GLOBAL_VARS["dataset_path"])
    if path_already_exists:
        print("Canceling...")
        return

    get_dataset_from_web(GLOBAL_VARS["dataset_url"], num_cut_dirs=num_cut_dirs)
    print()
    unpack_bz2_files(GLOBAL_VARS["dataset_path"])
    print()
    delete_bz2_files(GLOBAL_VARS["dataset_path"])


if __name__ == "__main__":
    main()
