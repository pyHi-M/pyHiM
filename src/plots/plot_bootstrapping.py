#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:42:39 2023

@author: marcnol

This script will perform bootstrapping analysis on PWD distance matrices

INPUTS:
    - PWD single cell maps in npy
    - uniquebarcode list


Example:
$ plot_bootstrapping.py --input Trace_Trace_all_ROIs_filtered_beta_mask0_exp_ND_1_PDX1LR_Matrix_PWDscMatrix.npy -U Trace_Trace_all_ROIs_filtered_exocrine_mask0_exp_ND_3_PDX1LR_Matrix_uniqueBarcodes.ecsv --N_bootstrap 100

Options:
    - plottingFileExtension: format of figure
    - cScale: value of the max of the cScale used to plot the matrix
    - cmap: name of cmap
    - scalingParameter: Normalizing scaling parameter of colormap. Max will matrix.max()/scalingParameter. Default is 1.
    - outputFolder: name of outputfolder. 'plots' is the default

"""


import argparse

# %% imports and plotting settings
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from plotting_functions import bootstraps_matrix, gets_matrix, plot_2d_matrix_simple

# define and loads datasets


def parse_arguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-T", "--input", help="Filename of single-cell PWD matrices in Numpy"
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
    parser.add_argument("--cMin_std", help="Colormap min scale for std map. Default: 0")
    parser.add_argument(
        "--cMax_std", help="Colormap max scale for std map. Default: automatic"
    )
    parser.add_argument(
        "--plottingFileExtension", help="By default: png. Other options: svg, pdf, png"
    )
    parser.add_argument(
        "--shuffle",
        help="Provide shuffle vector: 0,1,2,3... of the same size or smaller than the original matrix. No spaces! comma-separated!",
    )
    parser.add_argument("--cmap", help="Colormap. Default: coolwarm")
    parser.add_argument("--cmap_std", help="Colormap for std map. Default: coolwarm")

    parser.add_argument(
        "--N_bootstrap", help="Number of bootstrapping cycles. Default=9999"
    )

    args = parser.parse_args()

    run_parameters = {}

    if args.input:
        run_parameters["scPWDMatrix_filename"] = args.input
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
        run_parameters["fontsize"] = 9

    if args.axisLabel:
        run_parameters["axisLabel"] = args.axisLabel
    else:
        run_parameters["axisLabel"] = False

    if args.cMax:
        run_parameters["cMax"] = float(args.cMax)
    else:
        run_parameters["cMax"] = 0.0

    if args.cMin:
        run_parameters["cMin"] = float(args.cMin)
    else:
        run_parameters["cMin"] = -1

    if args.cMax_std:
        run_parameters["cMax_std"] = float(args.cMax_std)
    else:
        run_parameters["cMax_std"] = 0.0

    if args.cMin_std:
        run_parameters["cMin_std"] = float(args.cMin_std)
    else:
        run_parameters["cMin_std"] = -1

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".png"

    if args.shuffle:
        run_parameters["shuffle"] = args.shuffle
    else:
        run_parameters["shuffle"] = 0

    run_parameters["pixelSize"] = 1

    if args.N_bootstrap:
        run_parameters["N_bootstrap"] = int(args.N_bootstrap)
    else:
        run_parameters["N_bootstrap"] = 9999

    if args.cmap:
        run_parameters["cmap"] = args.cmap
    else:
        run_parameters["cmap"] = "RdBu"

    if args.cmap_std:
        run_parameters["cmap_std"] = args.cmap_std
    else:
        run_parameters["cmap_std"] = "Reds"

    run_parameters["dist_calc_mode"] = "median"
    run_parameters["scalingParameter"] = 1
    run_parameters["matrix_norm_mode"] = ""

    return run_parameters


def plot_results(
    matrix,
    run_parameters,
    uniqueBarcodes,
    n_cells,
    outputFileName,
    fileNameEnding=".png",
    c_max=0,
    c_min=-1,
    fig_title="bootstrapping_PWD_median",
    cmap="RdBu",
):
    fig1 = plt.figure(constrained_layout=True, figsize=(6, 6))
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f_1 = fig1.add_subplot(spec1[0, 0])  # 16

    axisLabel = True
    cmtitle = "distance, um"
    fontsize = run_parameters["fontsize"]
    axis_ticks = True

    if c_min == -1:
        c_min = np.nanmin(matrix)
    if c_max == 0:
        c_max = np.nanmax(matrix)

    plot_2d_matrix_simple(
        f_1,
        matrix,
        uniqueBarcodes,
        yticks=axisLabel,
        xticks=axisLabel,
        cmtitle=cmtitle,
        fig_title=fig_title,
        c_min=c_min,
        c_max=c_max,  # log10(0.05) = -1.3
        fontsize=fontsize,
        colorbar=True,
        axis_ticks=axis_ticks,
        c_m=cmap,
        show_title=True,
        n_cells=n_cells,
        n_datasets=2,
    )
    print("Output data: {}.npy".format(outputFileName + fig_title))
    np.save(outputFileName + fig_title, matrix)

    # saves output matrix in NPY format
    outputFileName = outputFileName + fig_title + fileNameEnding
    plt.savefig(outputFileName + fileNameEnding)
    print("Output figure: {}".format(outputFileName))


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
    print(f"$ cmap scales:\nmean:{run_parameters['cMin']}-{run_parameters['cMax']}")
    print(f"std:{run_parameters['cMin_std']}-{run_parameters['cMax_std']}\n")

    mean_bs, mean_error = bootstraps_matrix(
        sc_matrix, N_bootstrap=run_parameters["N_bootstrap"]
    )
    print(f"$ mean_bs = {mean_bs.shape}")

    plot_results(
        mean_bs,
        run_parameters,
        uniqueBarcodes,
        n_cells,
        outputFileName,
        c_max=run_parameters["cMax"],
        c_min=run_parameters["cMin"],
        fileNameEnding=run_parameters["plottingFileExtension"],
        fig_title="bootstrapping_median",
        cmap=run_parameters["cmap"],
    )

    plot_results(
        mean_error,
        run_parameters,
        uniqueBarcodes,
        n_cells,
        outputFileName,
        c_max=run_parameters["cMax_std"],
        c_min=run_parameters["cMin_std"],
        fileNameEnding=run_parameters["plottingFileExtension"],
        fig_title="bootstrapping_std_median",
        cmap=run_parameters["cmap_std"],
    )

    print("\nDone\n\n")


if __name__ == "__main__":
    main()
