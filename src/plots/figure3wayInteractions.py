#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:18:15 2020

@author: marcnol
"""


import argparse

# %% imports and plotting settings
import os

import matplotlib.gridspec as gridspec

# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

from matrixOperations.HIMmatrixOperations import AnalysisHiMMatrix

# import scaleogram as scg


# %% define and loads datasets
def parse_arguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument("-F1", "--rootFolder1", help="Folder with dataset 1")
    parser.add_argument("-F2", "--rootFolder2", help="Folder with dataset 2")
    parser.add_argument("-O", "--outputFolder", help="Folder for outputs")
    parser.add_argument(
        "-P",
        "--parameters",
        help="Provide name of parameter files. folders_to_load.json assumed as default",
    )
    parser.add_argument(
        "-P2",
        "--parameters2",
        help="Provide name of parameter files for dataset 2. folders_to_load.json assumed as default",
    )
    parser.add_argument(
        "-A1", "--label1", help="Add name of label for dataset 1 (e.g. doc)"
    )
    parser.add_argument(
        "-W1",
        "--action1",
        help="Select: [all], [labeled] or [unlabeled] cells plotted for dataset 1 ",
    )
    parser.add_argument(
        "-A2", "--label2", help="Add name of label for dataset 1  (e.g. doc)"
    )
    parser.add_argument(
        "-W2",
        "--action2",
        help="Select: [all], [labeled] or [unlabeled] cells plotted for dataset 1 ",
    )
    parser.add_argument("--fontsize", help="Size of fonts to be used in matrix")
    # parser.add_argument("--axisLabel", help="Use if you want a label in x and y", action="store_true")
    # parser.add_argument("--axisTicks", help="Use if you want axes ticks", action="store_true")
    parser.add_argument("--scalingParameter", help="Scaling parameter of colormap")
    parser.add_argument(
        "--colorbar", help="Use if you want a colorbar", action="store_true"
    )
    parser.add_argument(
        "--plottingFileExtension", help="By default: svg. Other options: pdf, png"
    )
    parser.add_argument(
        "--normalize",
        help="Matrices get normalized by their maximum",
        action="store_true",
    )
    args = parser.parse_args()

    run_parameters = {}
    run_parameters["pixelSize"] = 0.1

    if args.rootFolder1:
        rootFolder1 = args.rootFolder1
    else:
        rootFolder1 = "."

    if args.rootFolder2:
        rootFolder2 = args.rootFolder2
        run_parameters["run2Datasets"] = True
    else:
        rootFolder2 = "."
        run_parameters["run2Datasets"] = False

    if args.outputFolder:
        output_folder = args.outputFolder
    else:
        output_folder = "none"

    if args.parameters:
        run_parameters["parametersFileName"] = args.parameters
    else:
        run_parameters["parametersFileName"] = "folders_to_load.json"

    if args.parameters2:
        run_parameters["parametersFileName2"] = args.parameters2
    else:
        run_parameters["parametersFileName2"] = "folders_to_load.json"

    if args.label1:
        run_parameters["label1"] = args.label1
    else:
        run_parameters["label1"] = "doc"

    if args.label2:
        run_parameters["label2"] = args.label2
    else:
        run_parameters["label2"] = "doc"

    if args.action1:
        run_parameters["action1"] = args.action1
    else:
        run_parameters["action1"] = "labeled"

    if args.action2:
        run_parameters["action2"] = args.action2
    else:
        run_parameters["action2"] = "unlabeled"

    if args.fontsize:
        run_parameters["fontsize"] = args.fontsize
    else:
        run_parameters["fontsize"] = 12

    if args.scalingParameter:
        run_parameters["scalingParameter"] = float(args.scalingParameter)
    else:
        run_parameters["scalingParameter"] = 1.0

    if args.colorbar:
        run_parameters["colorbar"] = args.colorbar
    else:
        run_parameters["colorbar"] = False

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".svg"

    if args.normalize:
        run_parameters["normalize"] = args.normalize
    else:
        run_parameters["normalize"] = False

    return rootFolder1, rootFolder2, output_folder, run_parameters


# =============================================================================
# MAIN
# =============================================================================


def main():
    print(">>> Producing HiM 3-way matrices")

    # [parsing arguments
    rootFolder1, rootFolder2, output_folder, run_parameters = parse_arguments()

    print(">>> Loading first dataset from {}".format(rootFolder1))
    him_data_1 = AnalysisHiMMatrix(run_parameters, rootFolder1)
    him_data_1.run_parameters["action"] = him_data_1.run_parameters["action1"]
    him_data_1.run_parameters["label"] = him_data_1.run_parameters["label1"]
    him_data_1.load_data()

    if output_folder == "none":
        output_folder = him_data_1.data_folder

    output_filename = (
        output_folder
        + os.sep
        + "Fig_3wayContacts"
        + "_dataset1:"
        + him_data_1.dataset_name
        + "_label1:"
        + run_parameters["label1"]
        + "_action1:"
        + run_parameters["action1"]
    )

    if run_parameters["run2Datasets"]:
        print(">>> Loading second dataset from {}".format(rootFolder2))
        him_data_2 = AnalysisHiMMatrix(run_parameters, rootFolder2)
        him_data_2.run_parameters["action"] = him_data_2.run_parameters["action2"]
        him_data_2.run_parameters["label"] = him_data_2.run_parameters["label2"]
        him_data_2.run_parameters["parametersFileName"] = him_data_2.run_parameters[
            "parametersFileName2"
        ]
        him_data_2.load_data()

        output_filename = (
            output_filename
            + "_dataset2:"
            + him_data_2.dataset_name
            + "_label2:"
            + run_parameters["label2"]
            + "_action2:"
            + run_parameters["action2"]
        )
    output_filename += run_parameters["plottingFileExtension"]

    # 3-way interaction matrices
    c_max = (
        him_data_1.data["ensembleContactProbability"].max()
        / run_parameters["scalingParameter"]
    )

    anchors = [
        int(i.split(":")[1])
        for i in list(him_data_1.data_files.keys())
        if "anchor" in i
    ]
    fig2 = plt.figure(constrained_layout=True)
    nCols = np.ceil(len(anchors) / 2).astype(int)
    nRows = 2
    spec2 = gridspec.GridSpec(ncols=nCols, nrows=nRows, figure=fig2)

    FigList, Yticks, Xticks = [], [], []
    for i_row in range(nRows):
        for i_col in range(nCols):
            FigList.append(fig2.add_subplot(spec2[i_row, i_col]))
            if i_row == nRows - 1:
                Xticks.append(False)
            else:
                Xticks.append(False)
            if i_col == 0:
                Yticks.append(False)
            else:
                Yticks.append(False)

    FigLabels = [i for i in list(him_data_1.data_files.keys()) if "anchor" in i]
    # print("FigList:{}".format(FigLabels))
    legendList = [False] * len(anchors)
    legendList[0] = True

    for ifigure, i_fig_label, iyticks, ixticks in zip(
        FigList, FigLabels, Xticks, Yticks
    ):
        if run_parameters["run2Datasets"]:
            # mixed matrices from 2 datasets
            if run_parameters["normalize"]:
                matrix = (
                    him_data_1.data[i_fig_label] / him_data_1.data[i_fig_label].max()
                )
            else:
                matrix = him_data_1.data[i_fig_label]

            for i in range(matrix.shape[0]):
                for j in range(0, i):
                    if run_parameters["normalize"]:
                        matrix[i, j] = (
                            him_data_2.data[i_fig_label][i, j]
                            / him_data_2.data[i_fig_label].max()
                        )
                    else:
                        matrix[i, j] = him_data_2.data[i_fig_label][i, j]
        else:
            # only one matrix
            matrix = him_data_1.data[i_fig_label]

        print("Dataset: {} | c_scale= {}-{}".format(i_fig_label, 0, c_max))
        f2_ax1_im = him_data_1.plot_2d_matrix_simple(
            ifigure,
            matrix,
            list(him_data_1.data["uniqueBarcodes"]),
            iyticks,
            ixticks,
            cmtitle="probability",
            c_min=0,
            c_max=c_max,
            fontsize=12,
        )

        matrixOutputFileName = output_filename + "_" + i_fig_label + ".npy"
        print("Saving Matrix: {}".format(matrixOutputFileName))
        np.save(matrixOutputFileName, matrix)

    if run_parameters["colorbar"]:
        cbar_ax = fig2.add_axes([0.995, 0.25, 0.02, 0.6])
        cbar = fig2.colorbar(f2_ax1_im, cax=cbar_ax, fraction=0.046, pad=0.04)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=12)

    plt.savefig(output_filename)
    print("Output figure: {}".format(output_filename))

    print("\nDone\n\n")


if __name__ == "__main__":
    main()
