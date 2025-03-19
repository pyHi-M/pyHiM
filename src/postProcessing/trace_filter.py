#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:57:36 2022

@author: marcnol

trace_filter.py - Chromatin Trace Filtering Utility
==================================================

Description:
------------
This script processes and filters chromatin trace files based on various criteria:
- Spatial coordinates (x, y, z)
- Minimum number of barcodes per trace
- Duplicate spot removal
- Barcode-specific filtering
- Label-based filtering
- Localization intensity thresholding

The script can process single files or multiple files via pipe input.

Arguments:
----------
Basic Arguments:
    --input             Path to input trace file (.ecsv format)
    --output            Tag to add to the output filename (default: "filtered")
    --pipe              Read trace file list from stdin (for batch processing)

Filtering Options:
    --N_barcodes        Minimum number of barcodes per trace (default: 2)
    --clean_spots       Remove spots with same UID and repeated barcodes within traces
    --remove_barcode    Comma-separated list of barcode IDs to remove (e.g., "1,2,3")

Coordinate Filtering:
    --x_min, --x_max    X-coordinate limits (default: 0 to infinity)
    --y_min, --y_max    Y-coordinate limits (default: 0 to infinity)
    --z_min, --z_max    Z-coordinate limits (default: 0 to infinity)

Label Filtering:
    --label             Keep only traces with this label
    --remove_label      When used with --label, removes traces with the label instead of keeping them

Intensity Filtering:
    --localization_file Path to localization table (.ecsv format)
    --intensity_min     Minimum intensity threshold for localizations

Outputs:
--------
1. Filtered trace file (.ecsv) with naming convention:
    [original_filename]_[output_tag]_[label_tag].ecsv

2. For intensity filtering:
    - Histogram plots of localization intensities (before and after filtering)

3. For duplicate barcode cleaning:
    - Statistics plots (.png) showing number of spots with same barcode per trace

Examples:
---------
# Basic filtering with spatial constraints and minimum barcode requirement
$ python trace_filter.py --input Trace.ecsv --z_min 4 --z_max 5 --y_max 175 --output 'zy_filtered' --N_barcodes 3

# Remove duplicate spots and specific barcodes
$ python trace_filter.py --input Trace.ecsv --clean_spots --remove_barcode 1,3,5

# Keep only traces with a specific label
$ python trace_filter.py --input Trace.ecsv --label "region1"

# Remove traces with a specific label
$ python trace_filter.py --input Trace.ecsv --label "region1" --remove_label

# Filter by localization intensity
$ python trace_filter.py --input Trace.ecsv --localization_file Localizations.ecsv --intensity_min 1000

# Process multiple files via pipe
$ ls *Trace.ecsv | python trace_filter.py --pipe --N_barcodes 3

"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import select
import sys

import numpy as np

from imageProcessing.localization_table import LocalizationTable
from matrixOperations.chromatin_trace_table import ChromatinTraceTable

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-O", "--output", help="Tag to add to the output file. Default = filtered"
    )

    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    parser.add_argument(
        "--clean_spots",
        help="removes both spots with same UID and barcodes repeated in a single trace",
        action="store_true",
    )

    parser.add_argument(
        "--remove_label",
        help="Use this argument to remove traces with the label provided",
        action="store_true",
    )

    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument("--N_barcodes", help="minimum_number_barcodes. Default = 2")

    parser.add_argument("--z_min", help="Z minimum for a localization. Default = 0")
    parser.add_argument(
        "--z_max", help="Z maximum for a localization. Default = np.inf"
    )
    parser.add_argument("--y_min", help="Y minimum for a localization. Default = 0")
    parser.add_argument(
        "--y_max", help="Y maximum for a localization. Default = np.inf"
    )
    parser.add_argument("--x_min", help="X minimum for a localization. Default = 0")
    parser.add_argument(
        "--x_max", help="X maximum for a localization. Default = np.inf"
    )
    parser.add_argument("--remove_barcode", help="name of barcode to remove")

    parser.add_argument(
        "--label", help="Select traces containing this label, removes all other traces."
    )

    parser.add_argument("--localization_file", help="Name of input localizations file.")
    parser.add_argument(
        "--intensity_min",
        type=float,
        help="Minimum intensity threshold for localizations.",
    )

    p = {}

    args = parser.parse_args()

    if args.output:
        p["output"] = args.output
    else:
        p["output"] = "filtered"

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.clean_spots:
        p["clean_spots"] = True
    else:
        p["clean_spots"] = False

    if args.remove_label:
        p["keep"] = False
    else:
        p["keep"] = True

    if args.N_barcodes:
        p["N_barcodes"] = int(args.N_barcodes)
    else:
        p["N_barcodes"] = 2

    if args.z_min:
        p["z_min"] = float(args.z_min)
    else:
        p["z_min"] = 0

    if args.z_max:
        p["z_max"] = float(args.z_max)
    else:
        p["z_max"] = np.inf

    if args.y_min:
        p["y_min"] = float(args.y_min)
    else:
        p["y_min"] = 0

    if args.y_max:
        p["y_max"] = float(args.y_max)
    else:
        p["y_max"] = np.inf

    if args.x_min:
        p["x_min"] = float(args.x_min)
    else:
        p["x_min"] = 0

    if args.x_max:
        p["x_max"] = float(args.x_max)
    else:
        p["x_max"] = np.inf

    if args.remove_barcode:
        p["remove_barcode"] = args.remove_barcode
    else:
        p["remove_barcode"] = None

    if args.label:
        p["label"] = args.label
    else:
        p["label"] = None

    if args.localization_file:
        p["localization_file"] = args.localization_file

    if args.intensity_min:
        p["intensity_min"] = args.intensity_min

    p["trace_files"] = []
    if args.pipe:
        p["pipe"] = True
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.0,
        )[0]:
            p["trace_files"] = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        p["pipe"] = False
        p["trace_files"] = [p["input"]]

    return p


def runtime(
    trace_files=[],
    N_barcodes=2,
    coor_limits=dict(),
    tag="filtered",
    remove_duplicate_spots=False,
    remove_barcode=None,
    label="",
    keep=True,
    localizations_file=None,
    intensity_min=0,
):
    # checks number of trace files
    if len(trace_files) < 1:
        print(
            "! Error: no trace file provided. Please either use pipe or the --input option to provide a filename."
        )
        return 0
    elif len(trace_files) == 1:
        print("\n$ trace files to process= {}".format(trace_files))
    else:
        print(
            "\n{} trace files to process= {}".format(
                len(trace_files), "\n".join(map(str, trace_files))
            )
        )

    if intensity_min and localizations_file:
        localization_table = LocalizationTable()
        localizations_data, _ = localization_table.load(localizations_file)
        print(f"$ Loaded localizations table with: {len(localizations_data)} rows")

        # Plot intensity distribution to help user choose a threshold
        intensities = [row["peak"] for row in localizations_data]
        output_file = localizations_file.split(".")[0]
        localization_table.plot_intensity_distribution(
            intensities, output_file=output_file + "_localization_intensities.png"
        )

    coors = ["x", "y", "z"]
    if len(trace_files) > 0:
        # iterates over traces
        for trace_file in trace_files:
            trace = ChromatinTraceTable()
            trace.initialize()
            comments = list()

            # reads new trace
            trace.load(trace_file)

            # remove duplicated spots
            if remove_duplicate_spots:
                trace.remove_duplicates()

            # filters trace by minimum number of barcodes
            if N_barcodes > 1:
                trace.filter_traces_by_n(minimum_number_barcodes=N_barcodes)
                comments.append("filt:N_barcodes>" + str(N_barcodes))

            # filters trace by coordinate
            for coor in coors:
                coor_min = coor_limits[coor + "_min"]
                coor_max = coor_limits[coor + "_max"]

                if coor_min > 0.0 or coor_max != np.inf:
                    trace.filter_traces_by_coordinate(
                        coor=coor,
                        coor_min=coor_min,
                        coor_max=coor_max,
                    )
                    comments.append("filt:{}<{}>{}".format(coor_min, coor, coor_max))

            # removes barcodes in traces where they are repeated
            if remove_duplicate_spots:
                trace.filter_repeated_barcodes(trace_file)

            # removes barcodes from a list provided by user
            if remove_barcode is not None:
                bc_list = remove_barcode.split(",")
                print(f"\n$ Removing barcodes: {bc_list}")
                for bc in bc_list:
                    trace.remove_barcode(bc)

            # removes localizations with low intensity
            if intensity_min and localizations_file:
                intensities_kept = trace.filter_by_intensity(
                    trace, localizations_data, intensity_min
                )
                output_file = trace_file.split(".")[0]
                localization_table.plot_intensity_distribution(
                    intensities_kept, output_file=f"{output_file}_filtered_intensities"
                )

            # defines output file name
            if label is not None:
                if keep:
                    trace.trace_keep_label(label)
                    file_tag = label
                else:
                    trace.trace_remove_label(label)
                    file_tag = "not:" + label

                # saves output trace
                outputfile = (
                    trace_file.split(".")[0] + "_" + tag + "_" + file_tag + ".ecsv"
                )
            else:
                outputfile = trace_file.split(".")[0] + "_" + tag + ".ecsv"

            trace.save(outputfile, trace.data, comments=", ".join(comments))

    else:
        print("No trace file found to process!")

    return len(trace_files)


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 10 + "Started execution" + "=" * 10)

    # [parsing arguments]
    p = parse_arguments()

    # [loops over lists of datafolders]
    n_traces_processed = runtime(
        trace_files=p["trace_files"],
        N_barcodes=p["N_barcodes"],
        coor_limits=p,
        tag=p["output"],
        remove_duplicate_spots=p["clean_spots"],
        remove_barcode=p["remove_barcode"],
        label=p["label"],
        keep=p["keep"],
        localizations_file=p["localization_file"],
        intensity_min=p["intensity_min"],
    )

    print(f"Processed <{n_traces_processed}> trace file(s)\n")
    print("=" * 9 + "Finished execution" + "=" * 9)


if __name__ == "__main__":
    main()
