#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 2022

@author: marcnol

This script will load a localizations table and analyze a number of properties such as:
    - number of detections per barcode

$ analyze_localizations.py

Planned features:
    export: localization_table_stats.csv

    provide the possibility to merge several input localization files and produce joint statistics

"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import select
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from imageProcessing.localization_table import LocalizationTable

font = {"weight": "normal", "size": 12}

matplotlib.rc("font", **font)

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images")
    # parser.add_argument( "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true" )

    p = {}

    args = parser.parse_args()
    if args.rootFolder:
        p["rootFolder"] = args.rootFolder
    else:
        p["rootFolder"] = "."

    p["localization_files"] = list()

    if select.select(
        [
            sys.stdin,
        ],
        [],
        [],
        0.0,
    )[0]:
        p["localization_files"] = [line.rstrip("\n") for line in sys.stdin]
    else:
        print("Nothing in stdin. Please provide list of localization files to process.")

    return p


def get_barcode_statistics(barcode_map, output_filename="localization_analysis.png"):
    """
    Function that calculates the
        - histogram of number of localizations per barcode


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
    localizations_by_barcode = barcode_map.group_by("Barcode #")

    trace_lengths = list()

    for sub_trace_table in localizations_by_barcode.groups:
        trace_lengths.append(len(sub_trace_table))

    distributions = [trace_lengths]
    axis_x_labels = [
        "number of localizations",
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
        axis.set_ylabel("Number of barcodes")
        axis.set_title(
            "n = "
            + str(len(distribution))
            + " | median = "
            + str(np.median(distribution))
        )

    plt.savefig(output_filename)


def get_number_localization_per_barcode(
    barcode_map, output_filename="localization_analysis.png"
):
    """
    Function that calculates the
        - number of localizations per barcode

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
    localizations_by_barcode = barcode_map.group_by("Barcode #")

    barcode_lengths = list()
    barcode_name = list()
    for sub_trace_table in localizations_by_barcode.groups:
        barcode_lengths.append(len(sub_trace_table))
        unique_barcode_name = list(set(sub_trace_table["Barcode #"]))
        barcode_name.append(str(unique_barcode_name[0]))

    print("Barcodes detected: \n{}".format(barcode_name))

    distributions = [barcode_lengths]
    axis_x_labels = [
        "barcode #",
    ]

    number_plots = len(distributions)

    fig = plt.figure(constrained_layout=True)
    im_size = 10
    fig.set_size_inches((im_size * number_plots, im_size))
    gs = fig.add_gridspec(1, number_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(number_plots)]

    for axis, distribution, xlabel in zip(axes, distributions, axis_x_labels):
        axis.bar(barcode_name, distribution, alpha=0.5)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of localizations")
        axis.set_title(
            "n = "
            + str(len(distribution))
            + " | median = "
            + str(np.median(distribution))
        )

    plt.savefig(output_filename)


def analyze_table(table, localization_file, barcode_map, unique_barcodes):
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

    print(f"$ Number of lines in localization table: {len(barcode_map)}.")

    localization_stats_file = [
        localization_file.split(".")[0],
        "localization_table_stats",
        ".png",
    ]
    print("\n$ Stats: {}", format("".join(localization_stats_file)))

    localizations_file = [localization_file.split(".")[0], "_XYZ_localizations", ".png"]
    print("\n$ XYZ Localizations: {}", format("".join(localizations_file)))

    localization_stats_perBarcode_file = [
        localization_file.split(".")[0],
        "_localization_stats_perBarcode",
        ".png",
    ]
    print(
        "\n$ localization stats per Barcode: {}",
        format("".join(localization_stats_perBarcode_file)),
    )

    table.plot_distribution_fluxes(barcode_map, localization_stats_file)
    table.plots_localizations(barcode_map, localizations_file)

    # get_barcode_statistics(barcode_map, "".join(localization_stats_perBarcode_file))

    get_number_localization_per_barcode(
        barcode_map, "".join(localization_stats_perBarcode_file)
    )


def process_localizations(folder, localization_files=list()):
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

    print(
        "\n{} trace files to process= {}".format(
            len(localization_files), "\n".join(map(str, localization_files))
        )
    )

    if len(localization_files) > 0:
        # iterates over traces in folder
        for localization_file in localization_files:
            table = LocalizationTable()

            # reads new trace
            barcode_map, unique_barcodes = table.load(localization_file)

            print(f"> Analyzing traces for {localization_file}")

            print(f"> Plotting traces for {localization_file}")

            analyze_table(table, localization_file, barcode_map, unique_barcodes)


# =============================================================================
# MAIN
# =============================================================================


def main():
    # [parsing arguments]
    p = parseArguments()

    # [loops over lists of datafolders]
    folder = p["rootFolder"]
    process_localizations(folder, localization_files=p["localization_files"])

    print("Finished execution")


if __name__ == "__main__":
    main()
