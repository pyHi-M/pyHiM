#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:50:54 2021

@author: marcnol

unzips HiM_run.tar.gz recursively

In the command line, run as
$ unzipHiM_run.py

to unzip all the directories recursively

"""
import argparse
import glob
import os

# =============================================================================
# MAIN
# =============================================================================


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images, default: .")
    parser.add_argument(
        "-R",
        "--recursive",
        help="One more depth of folders will be explored and zipped",
        action="store_true",
    )

    args = parser.parse_args()

    if args.rootFolder:
        root_folder_tempo = args.rootFolder
    else:
        root_folder_tempo = os.getcwd()

    if args.recursive:
        recursive = args.recursive
    else:
        recursive = False

    if recursive:
        allFiles = glob.glob(root_folder_tempo + "/*")
        root_folders = [x for x in allFiles if os.path.isdir(x)]
    else:
        root_folders = [root_folder_tempo]

    print(f"RootFolders: {root_folders}")
    for root_folder in root_folders:
        # opens tarfile
        os.chdir(root_folder)
        tar_filename = "HiMrun.tar.gz"
        if os.path.exists(root_folder + os.sep + tar_filename):
            print(f"Unzipping archive: {tar_filename}")
            tarcmd = "tar -xzvf " + tar_filename
            print(f"cmd> {tarcmd}")
            os.system(tarcmd)
        else:
            print(f"Nothing to unzip in: {root_folder + os.sep + tar_filename}")


if __name__ == "__main__":
    main()
