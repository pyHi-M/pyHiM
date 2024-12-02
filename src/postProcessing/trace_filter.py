#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:57:36 2022

@author: marcnol

This script will load a trace file and a filter traces based on a series of user arguments

--> Usage

$ trace_filter.py --input Trace.ecsv --z_min 4 --z_max 5 --y_max 175 --output 'zy_filtered' --N_barcodes 3 --clean_spots

will analyze 'Trace.ecsv' and remove spots with 4>z>5 amd z>175 and less than 3 barcodes

--clean_spots will remove barcode spots that are repeated within a trace

--remove_barcode will remove the barcode name provided. This needs to be an integer or a comma-separated integer

--> outputs

.ecsv trace table file.
.png files with stats of number of spots for the same barcode per trace [only if --clean_spots was used]

"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import os
import select
import sys
from datetime import datetime

import numpy as np

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
        help="remove barcode spots repeated in a single trace",
        action="store_true",
    )

    parser.add_argument(
        "--remove_label",
        help="Use this argument to remove traces with the label provided",
        action="store_true",
    )

    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument("--N_barcodes", help="minimum_number_barcodes. Default = 2")
    parser.add_argument(
        "--dist_max", help="Maximum distance threshold. Default = np.inf"
    )
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

    if args.dist_max:
        p["dist_max"] = float(args.dist_max)
    else:
        p["dist_max"] = np.inf

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
    dist_max=np.inf,
    label="",
    keep=True,
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

    coors = ["x", "y", "z"]
    if len(trace_files) > 0:
        # iterates over traces
        for trace_file in trace_files:
            trace = ChromatinTraceTable()
            trace.initialize()
            comments = list()

            # reads new trace
            trace.load(trace_file)

            # remove duplicated barcodes
            trace.remove_duplicates()

            # filters trace by minimum number of barcodes
            trace.filter_traces_by_n(minimum_number_barcodes=N_barcodes)
            comments.append("filt:N_barcodes>" + str(N_barcodes))

            # filters trace by coordinate
            for coor in coors:
                trace.filter_traces_by_coordinate(
                    coor=coor,
                    coor_min=coor_limits[coor + "_min"],
                    coor_max=coor_limits[coor + "_max"],
                )
                comments.append(
                    "filt:{}<{}>{}".format(
                        coor_limits[coor + "_min"], coor, coor_limits[coor + "_max"]
                    )
                )

            # removes barcodes in traces where they are repeated
            if remove_duplicate_spots:
                trace.filter_repeated_barcodes(trace_file)

            if remove_barcode is not None:
                bc_list = remove_barcode.split(",")
                for bc in bc_list:
                    trace.remove_barcode(bc)

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
            print(f"$ Saved output trace file at: {outputfile}")
    else:
        print("No trace file found to process!")

    return len(trace_files)


# =============================================================================
# MAIN
# =============================================================================


def main():
    begin_time = datetime.now()

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
        dist_max=p["dist_max"],
        label=p["label"],
        keep=p["keep"],
    )

    print(f"Processed <{n_traces_processed}> trace file(s)\n")
    print("=" * 9 + "Finished execution" + "=" * 9)


if __name__ == "__main__":
    main()
