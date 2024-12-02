#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:07:05 2022

@author: marcnol

This script will load a trace file and analyze a number of properties such as:
    - number of barcodes detected per trace
    - number of duplicated barcodes
    - trace Rg

$ trace_analyzer.py

output:

trace_stats.csv

trace_ID, number of barcodes, number of duplications, Rg,



"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import collections
import select
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matrixOperations.chromatin_trace_table import ChromatinTraceTable

font = {"weight": "normal", "size": 22}

matplotlib.rc("font", **font)

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images")
    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    p = {}

    args = parser.parse_args()
    if args.rootFolder:
        p["rootFolder"] = args.rootFolder
    else:
        p["rootFolder"] = "."

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

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


def get_xyz_statistics(trace, output_filename="test_coor.png"):
    """
    Function that calculates the
        - distribution of localizations in x y z

    Parameters
    ----------
    trace : TYPE
        Trace table in ASTROPY Table format.
    output_filename : TYPE, optional
        Output figure in PNG. The default is 'test.png'.

    Returns
    -------
    None.

    """
    coords = ["x", "y", "z"]

    fig = plt.figure(constrained_layout=True)
    im_size, number_plots = 10, 3
    fig.set_size_inches((im_size * number_plots, im_size))
    gs = fig.add_gridspec(1, number_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(number_plots)]

    for axis, coor in zip(axes, coords):
        print(f"$ processing coordinate: {coor}")
        coordinates = trace[coor].data
        axis.hist(coordinates, alpha=0.3, bins=20)
        axis.set_xlabel(coor)
        axis.set_ylabel("counts")
        axis.set_title(
            "n = "
            + str(len(coordinates))
            + " | median = "
            + str(np.median(coordinates))
        )

    plt.savefig(output_filename)


def get_barcode_statistics(trace, output_filename="test_barcodes.png"):
    """
    Function that calculates the
        - number of barcodes per trace
        - number of unique barcodes per trace
        - number of repeated barcodes per trace

    Parameters
    ----------
    trace : TYPE
        Trace table in ASTROPY Table format.
    output_filename : TYPE, optional
        Output figure in PNG. The default is 'test.png'.

    Returns
    -------
    None.

    """
    trace_by_ID = trace.group_by("Trace_ID")

    trace_lengths = list()
    trace_unique_barcodes = list()
    trace_repeated_barcodes = list()
    number_unique_barcodes = list()
    number_repeated_barcodes = list()

    for sub_trace_table in trace_by_ID.groups:
        trace_lengths.append(len(sub_trace_table))

        unique_barcodes = list(set(sub_trace_table["Barcode #"]))
        trace_unique_barcodes.append(unique_barcodes)
        number_unique_barcodes.append(len(unique_barcodes))

        repeated_barcodes = [
            item
            for item, count in collections.Counter(sub_trace_table["Barcode #"]).items()
            if count > 1
        ]
        trace_repeated_barcodes.append(repeated_barcodes)
        number_repeated_barcodes.append(len(repeated_barcodes))

    distributions = [trace_lengths, number_unique_barcodes, number_repeated_barcodes]
    axis_x_labels = [
        "number of barcodes",
        "number of unique barcodes",
        "number of repeated barcodes",
    ]
    number_plots = len(distributions)

    fig = plt.figure(constrained_layout=True)
    im_size = 10
    fig.set_size_inches((im_size * number_plots, im_size))
    gs = fig.add_gridspec(1, number_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(number_plots)]

    for axis, distribution, xlabel in zip(axes, distributions, axis_x_labels):
        axis.hist(distribution, alpha=0.3)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("counts")
        axis.set_title(
            "n = "
            + str(len(distribution))
            + " | median = "
            + str(np.median(distribution))
        )

    plt.savefig(output_filename)


def analyze_trace(trace, trace_file):
    """
    Launcher function that will perform different kinds of trace analyses

    Parameters
    ----------
    trace : ChromatinTraceTable Class
        Trace table, instance of the ChromatinTraceTable Class.
    trace_file : string
        file name of trace table in ecsv format.

    Returns
    -------
    None.

    """

    trace_table = trace.data

    print(f"$ Number of lines in trace: {len(trace_table)}")

    output_filename = [trace_file.split(".")[0], "_xyz_statistics", ".png"]
    get_xyz_statistics(trace_table, "".join(output_filename))

    output_filename = [trace_file.split(".")[0], "_trace_statistics", ".png"]
    get_barcode_statistics(trace_table, "".join(output_filename))

    # plots statistics of barcodes and saves in file
    collective_barcode_stats = trace.barcode_statistics(trace_table)
    trace.plots_barcode_statistics(
        collective_barcode_stats, file_name=trace_file + "_stats", kind="matrix"
    )


def process_traces(trace_files=list()):
    """
    Processes list of trace files and sends each to get analyzed individually

    Parameters
    ----------
    folder : TYPE
        DESCRIPTION.
    trace_files : TYPE, optional
        DESCRIPTION. The default is list().

    Returns
    -------
    None.

    """

    if len(trace_files) > 0:
        print(
            "\n{} trace files to process= {}".format(
                len(trace_files), "\n".join(map(str, trace_files))
            )
        )

        # iterates over traces in folder
        for trace_file in trace_files:
            trace = ChromatinTraceTable()
            trace.initialize()

            # reads new trace
            trace.load(trace_file)

            print(f"> Analyzing traces for {trace_file}")

            print(f"> Plotting traces for {trace_file}")
            trace.plots_traces([trace_file.split(".")[0], "_traces_XYZ", ".png"])

            analyze_trace(trace, trace_file)

    else:
        print(
            "! Error: did not find any trace file to analyze. Please provide one using --input or --pipe."
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    begin_time = datetime.now()

    # [parsing arguments]
    p = parseArguments()

    # [loops over lists of datafolders]
    process_traces(trace_files=p["trace_files"])

    print("Finished execution")


if __name__ == "__main__":
    main()
