#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:24:51 2020

@author: marcnol
"""


import argparse
import os
import sys

from plotting_functions import (
    gets_matrix,
    normalize_matrix,
    plot_matrix_difference,
    plot_mixed_matrix,
    plot_Wilcoxon_matrix,
)

from matrixOperations.HIMmatrixOperations import plot_matrix


def parse_arguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--outputFolder", help="Folder for outputs")
    parser.add_argument(
        "-T1", "--input1", help="Filename of single-cell PWD matrices in Numpy"
    )
    parser.add_argument(
        "-T2", "--input2", help="Filename of single-cell PWD matrices in Numpy"
    )
    parser.add_argument(
        "-U", "--uniqueBarcodes", help="csv file with list of unique barcodes"
    )

    parser.add_argument("--fontsize", help="Size of fonts to be used in matrix")
    parser.add_argument(
        "--axisLabel", help="Use if you want a label in x and y", action="store_true"
    )
    parser.add_argument(
        "--axisTicks", help="Use if you want axes ticks", action="store_true"
    )
    parser.add_argument(
        "--ratio",
        help="Does ratio between matrices. Default: difference",
        action="store_true",
    )
    parser.add_argument(
        "--plottingFileExtension", help="By default: svg. Other options: pdf, png"
    )
    parser.add_argument(
        "--normalize",
        help="Matrix normalization factor: maximum, none, single value (normalize 2nd matrix by), bin pair e.g. 1,2",
    )
    parser.add_argument("--pixelSize", help="pixelSize in microns")
    parser.add_argument(
        "--matrix_norm_mode",
        help="Matrix normalization mode. Can be n_cells (default) or nonNANs",
    )
    parser.add_argument(
        "--dist_calc_mode",
        help="Mode used to calculate the mean distance. Can be either 'median', 'KDE' or 'proximity'. Default: median",
    )
    parser.add_argument("--proximity_threshold", help="proximity threshold in um")

    parser.add_argument("--cMax", help="Colormap max scale. Default: automatic")
    parser.add_argument(
        "--scalingParameter",
        help="Scaling parameter. Default: 1",
    )
    parser.add_argument(
        "--shuffle",
        help="Provide shuffle vector: 0,1,2,3... of the same size or smaller than the original matrix. No spaces! comma-separated!",
    )

    parser.add_argument("--cMin", help="Colormap min cscale. Default: 0")
    parser.add_argument("--cmap", help="Colormap. Default: coolwarm")

    run_parameters = {}

    args = parser.parse_args()

    if args.input1:
        run_parameters["input1"] = args.input1
    else:
        print(
            ">> ERROR: you must provide a filename with the single cell PWD matrices in Numpy format"
        )
        sys.exit(-1)

    if args.input2:
        run_parameters["input2"] = args.input2
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

    if args.proximity_threshold:
        run_parameters["proximity_threshold"] = float(args.proximity_threshold)
    else:
        run_parameters["proximity_threshold"] = 0.25

    if args.cmap:
        run_parameters["cmap"] = args.cmap
    else:
        run_parameters["cmap"] = "coolwarm"

    if args.fontsize:
        run_parameters["fontsize"] = args.fontsize
    else:
        run_parameters["fontsize"] = 12

    if args.pixelSize:
        run_parameters["pixelSize"] = args.pixelSize
    else:
        run_parameters["pixelSize"] = 1

    if args.axisLabel:
        run_parameters["axisLabel"] = args.axisLabel
    else:
        run_parameters["axisLabel"] = True

    if args.axisTicks:
        run_parameters["axisTicks"] = args.axisTicks
    else:
        run_parameters["axisTicks"] = True

    if args.ratio:
        run_parameters["ratio"] = args.ratio
    else:
        run_parameters["ratio"] = False

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".svg"

    if args.normalize:
        run_parameters["normalize"] = args.normalize
    else:
        run_parameters["normalize"] = "none"

    if args.dist_calc_mode:
        run_parameters["dist_calc_mode"] = args.dist_calc_mode
    else:
        run_parameters["dist_calc_mode"] = "median"

    if run_parameters["dist_calc_mode"] == "proximity":
        run_parameters["cmtitle"] = "proximity frequency"
    else:
        run_parameters["cmtitle"] = "distance, um"

    if args.matrix_norm_mode:
        run_parameters["matrix_norm_mode"] = args.matrix_norm_mode
    else:
        run_parameters["matrix_norm_mode"] = (
            "n_cells"  # norm: n_cells (default), nonNANs
        )

    if args.cMax:
        run_parameters["cMax"] = float(args.cMax)
    else:
        run_parameters["cMax"] = 0.0

    if args.scalingParameter:
        run_parameters["scalingParameter"] = args.scalingParameter
    else:
        run_parameters["scalingParameter"] = 1

    if args.shuffle:
        run_parameters["shuffle"] = args.shuffle
    else:
        run_parameters["shuffle"] = 0

    if args.cMin:
        run_parameters["cMin"] = float(args.cMin)
    else:
        run_parameters["cMin"] = 0.0

    print("Input parameters:{}".format(run_parameters))

    return run_parameters


def gets_ensemble_matrix(run_parameters, scPWDMatrix_filename=""):
    (
        sc_matrix,
        uniqueBarcodes,
        cScale,
        n_cells,
        outputFileName,
        fileNameEnding,
    ) = gets_matrix(
        run_parameters,
        scPWDMatrix_filename=scPWDMatrix_filename,
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

    return (
        sc_matrix,
        meansc_matrix,
        uniqueBarcodes,
        cScale,
        n_cells,
        outputFileName,
        fileNameEnding,
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    run_parameters = parse_arguments()
    print("*" * 50)
    # creates output folder
    if not os.path.exists(run_parameters["outputFolder"]):
        os.mkdir(run_parameters["outputFolder"])
        print("Folder created: {}".format(run_parameters["outputFolder"]))

    # loads matrices
    (
        m1_sc,
        m1,
        uniqueBarcodes,
        cScale1,
        n_cells1,
        outputFileName1,
        fileNameEnding,
    ) = gets_ensemble_matrix(
        run_parameters, scPWDMatrix_filename=run_parameters["input1"]
    )
    print("-" * 50)

    (
        m2_sc,
        m2,
        uniqueBarcodes,
        cScale2,
        n_cells2,
        outputFileName2,
        fileNameEnding,
    ) = gets_ensemble_matrix(
        run_parameters, scPWDMatrix_filename=run_parameters["input2"]
    )

    # plots results
    if m1.shape == m2.shape:
        plot_matrix_difference(
            m1,
            m2,
            uniqueBarcodes,
            normalize=run_parameters["normalize"],
            ratio=run_parameters["ratio"],
            c_scale=run_parameters["cMax"],
            axisLabel=run_parameters["axisLabel"],
            fontsize=run_parameters["fontsize"],
            axis_ticks=run_parameters["axisTicks"],
            outputFileName=outputFileName1,
            fig_title=run_parameters["cmtitle"],
            plottingFileExtension=run_parameters["plottingFileExtension"],
            n_cells=n_cells1 + n_cells2,
            cmap=run_parameters["cmap"],
        )

        plot_mixed_matrix(
            m1,
            m2,
            uniqueBarcodes,
            normalize=run_parameters["normalize"],
            axisLabel=run_parameters["axisLabel"],
            fontsize=run_parameters["fontsize"],
            axis_ticks=run_parameters["axisTicks"],
            cAxis=run_parameters["cMax"],
            outputFileName=outputFileName1,
            fig_title=run_parameters["cmtitle"],
            plottingFileExtension=run_parameters["plottingFileExtension"],
            n_cells=n_cells1 + n_cells2,
            cmap=run_parameters["cmap"],
        )

        if "proximity" not in run_parameters["dist_calc_mode"]:
            if "none" in run_parameters["normalize"]:  # sets default operation
                mode = "none"
            else:
                mode = run_parameters["normalize"]

            m1, m2, m2_norm = normalize_matrix(m1, m2, mode)

            plot_Wilcoxon_matrix(
                m1_sc,
                m2_sc,
                uniqueBarcodes,
                normalize=m2_norm,
                axisLabel=run_parameters["axisLabel"],
                fontsize=run_parameters["fontsize"],
                axis_ticks=run_parameters["axisTicks"],
                outputFileName=outputFileName1,
                fig_title=run_parameters["cmtitle"],
                plottingFileExtension=run_parameters["plottingFileExtension"],
                n_cells=n_cells1 + n_cells2,
                cmap=run_parameters["cmap"],
            )

    else:
        print("Error: matrices do not have the same dimensions!")


if __name__ == "__main__":
    main()
