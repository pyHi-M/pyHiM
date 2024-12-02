#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:59:34 2020

@author: marcnol

plots 4M profiles given a list of anchors.

Can work with up to two datasets


"""


import argparse

# %% imports and plotting settings
import os

import matplotlib.gridspec as gridspec

# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

from matrixOperations.HIMmatrixOperations import (
    AnalysisHiMMatrix,
    plot_1d_profile2datasets,
)

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
    parser.add_argument(
        "--axisLabel", help="Use if you want a label in x and y", action="store_true"
    )
    parser.add_argument(
        "--axisTicks", help="Use if you want axes ticks", action="store_true"
    )
    parser.add_argument(
        "--splines",
        help="Use if you want plot data using spline interpolations",
        action="store_true",
    )
    parser.add_argument("--cAxis", help="absolute cAxis value for colormap")
    parser.add_argument(
        "--plottingFileExtension", help="By default: svg. Other options: pdf, png"
    )
    parser.add_argument(
        "--legend", help="Use if you want to show legends", action="store_true"
    )
    parser.add_argument(
        "--normalize",
        help="Matrix normalization factor: maximum, none, single value (normalize 2nd profile by)",
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

    if args.axisLabel:
        run_parameters["axisLabel"] = args.axisLabel
    else:
        run_parameters["axisLabel"] = False

    if args.axisTicks:
        run_parameters["axisTicks"] = args.axisTicks
    else:
        run_parameters["axisTicks"] = True

    if args.splines:
        run_parameters["splines"] = args.splines
    else:
        run_parameters["splines"] = False

    if args.cAxis:
        run_parameters["cAxis"] = float(args.cAxis)
    else:
        run_parameters["cAxis"] = 0.8

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".svg"

    if args.legend:
        run_parameters["legend"] = args.legend
    else:
        run_parameters["legend"] = False

    if args.normalize:
        run_parameters["normalize"] = args.normalize
    else:
        run_parameters["normalize"] = "none"

    print("Input Folders:{}, {}".format(rootFolder1, rootFolder2))
    print("Input parameters:{}".format(run_parameters))

    return rootFolder1, rootFolder2, output_folder, run_parameters


# =============================================================================
# MAIN
# =============================================================================


def main():
    run2Datasets = False

    rootFolder1, rootFolder2, output_folder, run_parameters = parse_arguments()
    print("RootFolders: \n{}\n{}".format(rootFolder1, rootFolder2))
    him_data_1 = AnalysisHiMMatrix(run_parameters, rootFolder1)
    him_data_1.run_parameters["action"] = him_data_1.run_parameters["action1"]
    him_data_1.run_parameters["label"] = him_data_1.run_parameters["label1"]
    him_data_1.load_data()

    if output_folder == "none":
        output_folder = him_data_1.data_folder

    output_filename = (
        output_folder
        + os.sep
        + "Fig_4Mcontacts"
        + "_dataset1:"
        + him_data_1.dataset_name
        + "_label1:"
        + run_parameters["label1"]
        + "_action1:"
        + run_parameters["action1"]
    )

    if run_parameters["run2Datasets"]:
        him_data_2 = AnalysisHiMMatrix(run_parameters, rootFolder2)
        him_data_2.run_parameters["action"] = him_data_2.run_parameters["action2"]
        him_data_2.run_parameters["label"] = him_data_2.run_parameters["label2"]
        him_data_2.load_data()

        run2Datasets = True
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

    anchors = him_data_1.list_data[him_data_1.dataset_name]["3wayContacts_anchors"]
    print("Anchors: {}".format(anchors))

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
                Yticks.append(True)
            else:
                Yticks.append(False)

    FigLabels = [i for i in list(him_data_1.data_files.keys()) if "anchor" in i]
    legendList = [False] * len(anchors)
    if run_parameters["legend"]:
        legendList[0] = True

    for anchor, ifigure, i_fig_label, yticks, xticks, legend in zip(
        anchors, FigList, FigLabels, Yticks, Xticks, legendList
    ):
        if not run2Datasets:
            him_data_1.plot_1d_profile1dataset(
                ifigure, anchor, i_fig_label, yticks, xticks
            )
        else:
            plot_1d_profile2datasets(
                ifigure,
                him_data_1,
                him_data_2,
                run_parameters,
                anchor,
                i_fig_label,
                yticks,
                xticks,
                legend,
            )

    plt.savefig(output_filename)
    print("Output figure: {}".format(output_filename))


if __name__ == "__main__":
    main()
