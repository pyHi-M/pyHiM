#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:04:10 2020
edited on Sep 6 2022

This script calculates and plots matrices (PWD and proximity) from:
    - a file with single-cell PWD matrices in Numpy format
    - a file with the unique barcodes used


Example:
$ figureHiMmatrix.py -T Trace_3D_barcode_KDtree_ROI:4_Matrix_PWDsc_matrix.npy -U Trace_3D_barcode_KDtree_ROI:4_Matrix_uniqueBarcodes.ecsv

Options:
    - plottingFileExtension: format of figure
    - cScale: value of the max of the cScale used to plot the matrix
    - cmap: name of cmap
    - scalingParameter: Normalizing scaling parameter of colormap. Max will matrix.max()/scalingParameter. Default is 1.
    - mode: indicated the plotting mode, either ["proximity"] or ["KDE", "median"] for PWD matrix.
    - outputFolder: name of outputfolder. 'plots' is the default

Left to do:
    - need to implement a way to select a subset of chromatin traces...

@author: marcnol
"""


import argparse
import os
import sys

import numpy as np
from plotting_functions import gets_matrix

from matrixOperations.HIMmatrixOperations import plot_matrix


def parse_arguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-T", "--scPWDMatrix", help="Filename of single-cell PWD matrices in Numpy"
    )
    parser.add_argument(
        "-U", "--uniqueBarcodes", help="csv file with list of unique barcodes"
    )
    parser.add_argument("-O", "--outputFolder", help="Folder for outputs")

    parser.add_argument("--fontsize", help="Size of fonts to be used in matrix")
    parser.add_argument(
        "--axisLabel", help="Use if you want a label in x and y", action="store_true"
    )
    parser.add_argument("--cMin", help="Colormap min scale. Default: 0")
    parser.add_argument("--cMax", help="Colormap max scale. Default: automatic")
    parser.add_argument(
        "--plottingFileExtension", help="By default: png. Other options: svg, pdf, png"
    )
    parser.add_argument(
        "--shuffle",
        help="Provide shuffle vector: 0,1,2,3... of the same size or smaller than the original matrix. No spaces! comma-separated!",
    )
    parser.add_argument("--proximity_threshold", help="proximity threshold in um")
    parser.add_argument("--cmap", help="Colormap. Default: coolwarm")
    parser.add_argument(
        "--dist_calc_mode",
        help="Mode used to calculate the mean distance. Can be either 'median', 'KDE' or 'proximity'. Default: median",
    )
    parser.add_argument(
        "--matrix_norm_mode",
        help="Matrix normalization mode. Can be n_cells or nonNANs (default)",
    )
    parser.add_argument(
        "--scalingParameter",
        help="Scaling parameter. Default: 1",
    )

    args = parser.parse_args()

    run_parameters = {}

    if args.scPWDMatrix:
        run_parameters["scPWDMatrix_filename"] = args.scPWDMatrix
    else:
        print(
            ">> ERROR: you must provide a filename with the single cell PWD matrices in Numpy format"
        )
        sys.exit(-1)

    if args.uniqueBarcodes:
        run_parameters["uniqueBarcodes"] = args.uniqueBarcodes
    else:
        print(">> ERROR: you must provide a CSV file with the unique barcodes used")
        sys.exit(-1)

    if args.outputFolder:
        run_parameters["outputFolder"] = args.outputFolder
    else:
        run_parameters["outputFolder"] = "plots"

    if args.fontsize:
        run_parameters["fontsize"] = args.fontsize
    else:
        run_parameters["fontsize"] = 22

    if args.axisLabel:
        run_parameters["axisLabel"] = args.axisLabel
    else:
        run_parameters["axisLabel"] = False

    if args.scalingParameter:
        run_parameters["scalingParameter"] = float(args.scalingParameter)
    else:
        run_parameters["scalingParameter"] = 1.0

    if args.proximity_threshold:
        run_parameters["proximity_threshold"] = float(args.proximity_threshold)
    else:
        run_parameters["proximity_threshold"] = 0.25

    if args.cMax:
        run_parameters["cMax"] = float(args.cMax)
    else:
        run_parameters["cMax"] = 0.0

    if args.cMin:
        run_parameters["cMin"] = float(args.cMin)
    else:
        run_parameters["cMin"] = 0.0

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".png"

    if args.shuffle:
        run_parameters["shuffle"] = args.shuffle
    else:
        run_parameters["shuffle"] = 0

    run_parameters["pixelSize"] = 1

    if args.cmap:
        run_parameters["cmap"] = args.cmap
    else:
        run_parameters["cmap"] = "coolwarm"

    if args.dist_calc_mode:
        run_parameters["dist_calc_mode"] = args.dist_calc_mode
    else:
        run_parameters["dist_calc_mode"] = "KDE"

    if run_parameters["dist_calc_mode"] == "proximity":
        run_parameters["cmtitle"] = "proximity frequency"
    else:
        run_parameters["cmtitle"] = "distance, um"

    if args.matrix_norm_mode:
        run_parameters["matrix_norm_mode"] = args.matrix_norm_mode
    else:
        run_parameters["matrix_norm_mode"] = (
            "nonNANs"  # norm: n_cells (default), nonNANs
        )

    if args.scalingParameter:
        run_parameters["scalingParameter"] = args.scalingParameter
    else:
        run_parameters["scalingParameter"] = 1

    return run_parameters


# %%

# =============================================================================
# MAIN
# =============================================================================


def main():
    print(">>> Producing HiM matrix")

    run_parameters = parse_arguments()

    if not os.path.exists(run_parameters["outputFolder"]):
        os.mkdir(run_parameters["outputFolder"])
        print("Folder created: {}".format(run_parameters["outputFolder"]))

    (
        sc_matrix,
        uniqueBarcodes,
        cScale,
        n_cells,
        outputFileName,
        fileNameEnding,
    ) = gets_matrix(
        run_parameters,
        scPWDMatrix_filename=run_parameters["scPWDMatrix_filename"],
        uniqueBarcodes=run_parameters["uniqueBarcodes"],
    )

    meansc_matrix = plot_matrix(
        sc_matrix,
        uniqueBarcodes,
        run_parameters["pixelSize"],
        1,
        outputFileName,
        "log",
        figtitle="Map: " + run_parameters["dist_calc_mode"],
        mode=run_parameters["dist_calc_mode"],  # median or KDE
        clim=cScale,
        c_min=run_parameters["cMin"],
        n_cells=n_cells,
        c_m=run_parameters["cmap"],
        cmtitle=run_parameters["cmtitle"],
        filename_ending=fileNameEnding + run_parameters["plottingFileExtension"],
        font_size=run_parameters["fontsize"],
    )

    print("Output figure: {}".format(outputFileName))

    # saves output matrix in NPY format
    outputFileName = outputFileName + fileNameEnding
    np.save(outputFileName, meansc_matrix)
    print("Output data: {}.npy".format(outputFileName))

    print("\nDone\n\n")


if __name__ == "__main__":
    main()
