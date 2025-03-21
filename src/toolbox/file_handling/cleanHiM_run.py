#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:16:07 2020

@author: marcnol

cleans directory structure

In the command line, run as
$ cleanHiM_run.py

to erase all the directories produced recursively

and

$ cleanHiM_run.py --all

to erase also the output MD, Log, and Session files

"""
import argparse
import glob
import os
import shutil

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images, default: .")
    parser.add_argument(
        "-A", "--all", help="Deletes folders, MD files, LOG files", action="store_true"
    )

    args = parser.parse_args()

    if args.rootFolder:
        root_folder = args.rootFolder
    else:
        root_folder = "."

    if args.all:
        markdown_files = glob.glob(
            root_folder + os.sep + "HiM_analysis*.md", recursive=True
        )
        md_log_files = glob.glob(
            root_folder + os.sep + "HiM_analysis*.log", recursive=True
        )
        log_files = glob.glob(root_folder + os.sep + "log*.txt", recursive=True)
        session_files = glob.glob(
            root_folder + os.sep + "Session*.json", recursive=True
        )
        tmp_img = glob.glob(root_folder + os.sep + "tmp.png")
        il_model = glob.glob(root_folder + os.sep + "infoList_model.json")
        params_model = glob.glob(root_folder + os.sep + "parameters_model.json")
        params_model += glob.glob(root_folder + os.sep + "parameters_used.json")

        for f in (
            markdown_files
            + log_files
            + session_files
            + md_log_files
            + tmp_img
            + il_model
            + params_model
        ):
            try:
                os.remove(f)
                print(f"File deleted: {f} ")
            except OSError as e:
                print(f"Error: {f} : {e.strerror}")
    # Removes directories produced during previous runs
    folders_to_remove = []
    folders_to_remove.append(root_folder + os.sep + "zProject")
    folders_to_remove.append(root_folder + os.sep + "alignImages")
    folders_to_remove.append(root_folder + os.sep + "segmentedObjects")
    folders_to_remove.append(root_folder + os.sep + "buildsPWDmatrix")
    # Add new routine names
    folders_to_remove.append(root_folder + os.sep + "project")
    folders_to_remove.append(root_folder + os.sep + "register_global")
    folders_to_remove.append(root_folder + os.sep + "register_local")
    folders_to_remove.append(root_folder + os.sep + "mask_2d")
    folders_to_remove.append(root_folder + os.sep + "mask_3d")
    folders_to_remove.append(root_folder + os.sep + "localize_2d")
    folders_to_remove.append(root_folder + os.sep + "localize_3d")
    folders_to_remove.append(root_folder + os.sep + "tracing")

    for new_folder in folders_to_remove:
        if os.path.isdir(new_folder):
            shutil.rmtree(new_folder)
            print(f"{new_folder} removed")
        # else:
        #     print(f"{new_folder} does not exist")


if __name__ == "__main__":
    main()
