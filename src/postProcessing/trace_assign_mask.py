#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 10:47:29 2022

@author: marcnol

This script will load a trace file and a number of numpy masks and assign them labels

$ trace_selector.py

outputs

ChromatinTraceTable() object and output .ecsv trace table file .


"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import os
import select
import sys

import numpy as np

from imageProcessing.imageProcessing import Image
from matrixOperations.chromatin_trace_table import ChromatinTraceTable

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input trace file")
    parser.add_argument(
        "--mask_file", help="Input mask image file. Expected format: NPY"
    )
    parser.add_argument(
        "--pixel_size", help="Lateral pixel size un microns. Default = 0.1"
    )
    parser.add_argument("--label", help="Label to add to trace file. Default=labeled")

    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    p = {}

    args = parser.parse_args()
    p["trace_files"] = []

    if args.input:
        p["trace_files"].append(args.input)

    if args.mask_file:
        p["mask_file"] = args.mask_file
    else:
        print(">> ERROR: you must provide a filename with a mask file")
        sys.exit(-1)

    if args.pixel_size:
        p["pixel_size"] = args.pixel_size
    else:
        p["pixel_size"] = 0.1

    if args.label:
        p["label"] = args.label
    else:
        p["label"] = "labeled"

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

    if len(p["trace_files"]) < 1:
        print(">> ERROR: you must provide a filename with a trace file")
        sys.exit(-1)
    return p


def assign_masks(trace, mask_file, label="labeled", pixel_size=0.1):

    # [checks if mask file exists for the file to process]

    if os.path.exists(mask_file):

        # load mask
        mask = Image()
        mask.data_2d = np.load(mask_file, allow_pickle=False).squeeze()
        print(f"$ mask image file read: {mask_file}")

        # matches traces and masks
        index = 0
        labeled_trace = []
        for trace_row in trace.data:
            x_int = int(trace_row["x"] / pixel_size)
            y_int = int(trace_row["y"] / pixel_size)
            if "x" in trace_row["label"]:
                trace_row["label"] = "_"

            # labels are appended as comma separated lists. Thus a trace can have multiple labels
            if mask.data_2d[x_int, y_int] == 1:
                trace_row["label"] = trace_row["label"] + "," + label
                index += 1
                labeled_trace.append(trace_row["Trace_ID"])

        unique_traces_labeled = set(labeled_trace)
        print(
            f"\n> {index} trace rows out of {len(trace.data)} were associated to mask {label}. Unique traces: {len(unique_traces_labeled)}"
        )
    else:
        print(f"ERROR: No mask image file found with name: {mask_file}")
        sys.exit(-1)

    return trace


def process_traces(trace_files=[], mask_file="", label="labeled", pixel_size=0.1):

    print(
        "\n{} trace files to process= {}".format(
            len(trace_files), "\n".join(map(str, trace_files))
        )
    )

    if trace_files:
        # iterates over traces in folder
        for trace_file in trace_files:
            trace = ChromatinTraceTable()
            trace.initialize()

            # reads new trace
            trace.load(trace_file)

            trace = assign_masks(trace, mask_file, label=label, pixel_size=pixel_size)

            outputfile = trace_file.rstrip(".ecsv") + "_" + label + ".ecsv"

            trace.save(outputfile, trace.data, comments=label)

            print(f"$ Saved output trace file at: {outputfile}")


# =============================================================================
# MAIN
# =============================================================================


def main():

    # [parsing arguments]
    p = parse_arguments()

    print("=" * 10 + "Started execution" + "=" * 10)

    process_traces(
        trace_files=p["trace_files"],
        mask_file=p["mask_file"],
        label=p["label"],
        pixel_size=p["pixel_size"],
    )

    print("=" * 9 + "Finished execution" + "=" * 9)


if __name__ == "__main__":
    main()
