#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

localization_merge.py
=====================

Description:
-----------
A utility script for merging multiple localization files into a single consolidated output.
This script is designed to work with localization tables containing spatial coordinates
and related metadata for microscopy or imaging analysis.

The script takes localization files as input via stdin (e.g., through piping or redirection)
and merges them into a single output file. It preserves all data from the original files
while combining them into one comprehensive table.

Usage:
-----
    $ cat file_list.txt | python localization_merge.py [options]
    $ ls *.ecsv | python localization_merge.py [options]
    $ find . -name "*.ecsv" | python localization_merge.py [options]

Arguments:
---------
    -o, --output_file    Name of the output file (default: merged_localizations.ecsv)
    -O, --output_folder  Path to the output folder (default: current directory)

Examples:
--------
1. Merge all .ecsv files in the current directory and save to the default output:
   $ ls *.ecsv | python localization_merge.py

2. Merge specific files and save with a custom name:
    $ echo -e "file1.ecsv\nfile2.ecsv\nfile3.ecsv" | python localization_merge.py -o combined.ecsv

3. Merge files and save to a specific directory:
    $ find ./data -name "loc_*.ecsv" | python localization_merge.py -O ./results -o merged.ecsv

4. Process files listed in a text file:
    $ cat files_to_merge.txt | python localization_merge.py

Notes:
-----
- The script requires the LocalizationTable class from the imageProcessing module
- Input files must be in a format readable by the LocalizationTable.load() method
- Output is saved in ECSV (Enhanced Character Separated Values) format
- The script reports the number of localizations in each file and the final merged file

Author:
------
marcnol (Original: March 15, 2023)
"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import os
import select
import sys

from imageProcessing.localization_table import LocalizationTable

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output File name. Default = merged_localizations.ecsv",
    )
    parser.add_argument("-O", "--output_folder", help="Output File name. Default = ./")
    p = {}

    args = parser.parse_args()
    if args.output_folder:
        p["outputFolder"] = args.output_folder
    else:
        p["outputFolder"] = "."

    if args.output_file:
        p["output_file"] = args.output_file
    else:
        p["output_file"] = "merged_localizations.ecsv"

    p["loc_files"] = []
    if select.select(
        [
            sys.stdin,
        ],
        [],
        [],
        0.0,
    )[0]:
        p["loc_files"] = [line.rstrip("\n") for line in sys.stdin]
    else:
        print("Nothing in stdin!\n")

    print("Input parameters\n" + "-" * 15)
    for item in p.keys():
        print("{}-->{}".format(item, p[item]))

    return p


def appends_traces(loc_files):
    new_loc_table = LocalizationTable()

    number_loc_tables = 0

    # iterates over traces in folder
    for loc_file in loc_files:

        print(f"$ loc file to process: {loc_file}")

        # reads new trace
        new_table, _ = new_loc_table.load(loc_file)

        # adds it to existing trace collection
        if number_loc_tables == 0:
            collected_tables = new_table
        else:
            collected_tables = new_loc_table.append(collected_tables, new_table)
        number_loc_tables += 1

        print(f" $ appended loc file with {len(new_table)} localizations")

    print(f" $ Merged loc file will contain {len(collected_tables)} localizations")

    return collected_tables, number_loc_tables


def load_localizations(loc_files=[]):

    if len(loc_files) > 1:
        # user provided a list of files to concatenate
        collected_tables, number_loc_tables = appends_traces(loc_files)

    print(f"Read and accumulated {number_loc_tables} localization files")

    return collected_tables, number_loc_tables


def run(p):
    print("\n" + "-" * 80)
    localizations = LocalizationTable()

    # [ creates output folder]
    if not os.path.exists(p["outputFolder"]):
        os.mkdir(p["outputFolder"])
        print("Folder created: {}".format(p["outputFolder"]))

    # loads and merges traces
    collected_tables, number_loc_tables = load_localizations(loc_files=p["loc_files"])

    # saves merged trace table
    output_file = p["output_file"]

    localizations.save(
        output_file,
        collected_tables,
        comments="appended_loc_files=" + str(number_loc_tables),
    )

    print("Finished execution")


# =============================================================================
# MAIN
# =============================================================================


def main():
    # [parsing arguments]
    p = parse_arguments()

    print("loc_files{}".format(len(p["loc_files"])))

    if len(p["loc_files"]) < 1:
        print("\nNothing to process...\n")
    else:
        run(p)


if __name__ == "__main__":
    main()
