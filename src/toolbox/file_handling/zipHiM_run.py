#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:14:47 2020

@author: marcnol

zips all png, MD and output files from a HiM run. It excludes .npy and Tiff images.

In the command line, run as
$ zipHiM_run.py

to zip all the directories recursively

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
        root_folder = args.rootFolder
    else:
        root_folder = os.getcwd()

    if args.recursive:
        RECURSIVE = args.recursive
    else:
        RECURSIVE = False

    print(f"RootFolders: {root_folder}")

    # opens tarfile
    os.chdir(root_folder)
    TAR_FILENAME = "HiMrun.tar"
    print(f"creating archive: {TAR_FILENAME} in {root_folder}")

    # tar files in root_folder
    markdown_files = [
        os.path.basename(f) for f in glob.glob(root_folder + os.sep + "*.md")
    ]
    log_files = [os.path.basename(f) for f in glob.glob(root_folder + os.sep + "*.log")]
    session_files = [
        os.path.basename(f) for f in glob.glob(root_folder + os.sep + "*.json")
    ]

    TARCMD = (
        "tar -cvf "
        + TAR_FILENAME
        + " "
        + " ".join(markdown_files + log_files + session_files)
    )
    print(
        "Archiving:\n{}".format("\n".join(markdown_files + log_files + session_files))
    )

    os.system(TARCMD)

    # tars directories produced during previous runs
    if RECURSIVE:
        folders = glob.glob(root_folder + os.sep + "*")
        folders = [x for x in folders if os.path.isdir(x)]  # keeps only folders
        folders = [x for x in folders if os.path.exists(x + os.sep + "parameters.json")]

    else:
        folders = [root_folder]

    print(f"Folders to zip:\n{folders}")
    print("=" * 30)
    for current_folder in folders:
        folders2zip = []
        folders2zip.append(current_folder + os.sep + "zProject")
        folders2zip.append(current_folder + os.sep + "alignImages")
        folders2zip.append(current_folder + os.sep + "segmentedObjects")
        folders2zip.append(current_folder + os.sep + "buildsPWDmatrix")

        print(f"sub-folders to zip:\n{folders2zip}")

        for new_folder in folders2zip:
            if root_folder == ".":
                new_folder_relative = "." + new_folder.split(os.getcwd())[1]
            else:
                new_folder_relative = "." + new_folder.split(root_folder)[1]

            file_extensions = ["/*.png", "/*.dat", "/*.ecsv", "/buildsPWDmatrix*.npy"]
            for new_file_extensions in file_extensions:
                new_files = new_folder_relative + new_file_extensions

                if len(glob.glob(new_files)) > 0:
                    TARCMD = "tar -rf " + TAR_FILENAME + " " + new_files
                    os.system(TARCMD)
                    print(f"Archiving: {new_files}")

        print("-" * 30)

    if os.path.exists(TAR_FILENAME):
        print(f"Zipping {TAR_FILENAME}")
        os.system("gzip " + TAR_FILENAME)


if __name__ == "__main__":
    main()
