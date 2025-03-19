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


def plot_neighbor_distances(trace, output_filename="neighbor_distances.png"):
    """
    Calculates the mean and standard deviation of X, Y, and Z distances between strictly consecutive neighboring barcodes
    and plots histograms for each.

    Parameters
    ----------
    trace : ChromatinTraceTable Class
        Trace table, instance of the ChromatinTraceTable Class.
    output_filename : str, optional
        The filename for the output PNG figure. Default is "neighbor_distances.png".

    Returns
    -------
    mean_dx, mean_dy, mean_dz : float
        Mean X, Y, and Z distances between strictly consecutive neighboring barcodes.
    std_dx, std_dy, std_dz : float
        Standard deviations of X, Y, and Z distances.
    """

    trace_table = trace.data
    trace_by_ID = trace_table.group_by("Trace_ID")

    dx_all, dy_all, dz_all = [], [], []

    for sub_trace_table in trace_by_ID.groups:
        # Sort by Barcode #
        sorted_trace = sub_trace_table[np.argsort(sub_trace_table["Barcode #"])]

        # Get barcodes and coordinates
        barcodes = sorted_trace["Barcode #"].data
        x_coords = sorted_trace["x"].data
        y_coords = sorted_trace["y"].data
        z_coords = sorted_trace["z"].data

        # Iterate and calculate distances only for strictly consecutive barcodes
        for i in range(len(barcodes) - 1):
            if barcodes[i + 1] == barcodes[i] + 1:  # Ensure strict consecutiveness
                dx_all.append(x_coords[i + 1] - x_coords[i])
                dy_all.append(y_coords[i + 1] - y_coords[i])
                dz_all.append(z_coords[i + 1] - z_coords[i])

    # Compute mean and standard deviation
    mean_dx, std_dx = (np.mean(dx_all), np.std(dx_all)) if dx_all else (0, 0)
    mean_dy, std_dy = (np.mean(dy_all), np.std(dy_all)) if dy_all else (0, 0)
    mean_dz, std_dz = (np.mean(dz_all), np.std(dz_all)) if dz_all else (0, 0)

    # Create figure with three histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Histogram of Strictly Consecutive Neighboring Barcode Distances")

    data = [dx_all, dy_all, dz_all]
    labels = [r"$\Delta x$, um", r"$\Delta y$, um", r"$\Delta z$, um"]
    colors = ["blue", "green", "red"]
    means = [mean_dx, mean_dy, mean_dz]
    stds = [std_dx, std_dy, std_dz]

    for i, (ax, dist, label, mean_val, std_val, color) in enumerate(
        zip(axes, data, labels, means, stds, colors)
    ):
        ax.hist(dist, bins=30, alpha=0.7, color=color, edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Counts")
        ax.set_title(
            f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}", fontsize=10
        )  # Smaller title

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"$ Saved neighbor distances plot: {output_filename}")

    return mean_dx, mean_dy, mean_dz, std_dx, std_dy, std_dz


'''
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
'''


def analyze_trace(trace, trace_file):
    """
    Launcher function that will perform different kinds of trace analyses.

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

    # Compute and plot neighbor distances
    output_filename = [trace_file.split(".")[0], "_neighbor_distances", ".png"]
    mean_dx, mean_dy, mean_dz, std_dx, std_dy, std_dz = plot_neighbor_distances(
        trace, "".join(output_filename)
    )
    print(
        f"$ Mean distances between neighboring barcodes: X={mean_dx:.3f}, Y={mean_dy:.3f}, Z={mean_dz:.3f}"
    )

    # Plots statistics of barcodes and saves in file
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

            print(f"> Plotting traces for {trace_file}")
            trace.plots_traces(
                [trace_file.split(".")[0], "_traces_XYZ", ".png"],
                pixel_size=[0.1, 0.1, 0.25],
            )

            print(f"> Analyzing traces for {trace_file}")
            analyze_trace(trace, trace_file)

    else:
        print(
            "! Error: did not find any trace file to analyze. Please provide one using --input or --pipe."
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    # [parsing arguments]
    p = parseArguments()

    # [loops over lists of datafolders]
    process_traces(trace_files=p["trace_files"])

    print("Finished execution")


if __name__ == "__main__":
    main()
