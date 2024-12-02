#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:01:30 2020

plots N Hi-M matrices in a subplot
@author: marcnol
"""


import argparse
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from matrixOperations.HIMmatrixOperations import (
    AnalysisHiMMatrix,
    calculate_contact_probability_matrix,
    list_sc_to_keep,
    shuffle_matrix,
)


def parse_arguments():
    # [parsing arguments]
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with dataset")
    parser.add_argument("-O", "--outputFolder", help="Folder for outputs")

    parser.add_argument(
        "-P",
        "--parameters",
        help="Provide name of parameter files. folders_to_load.json assumed as default",
    )
    parser.add_argument("-A", "--label", help="Add name of label (e.g. doc)")
    parser.add_argument(
        "-W", "--action", help="Select: [all], [labeled] or [unlabeled] cells plotted "
    )
    parser.add_argument("--fontsize", help="Size of fonts to be used in matrix")
    parser.add_argument(
        "--axisLabel", help="Use if you want a label in x and y", action="store_true"
    )
    parser.add_argument(
        "--axisTicks", help="Use if you want axes ticks", action="store_true"
    )
    parser.add_argument(
        "--barcodes",
        help="Use if you want barcode images to be displayed",
        action="store_true",
    )
    parser.add_argument("--scalingParameter", help="Scaling parameter of colormap")
    parser.add_argument(
        "--plottingFileExtension", help="By default: svg. Other options: pdf, png"
    )
    parser.add_argument(
        "--shuffle",
        help="Provide shuffle vector: 0,1,2,3... of the same size or smaller than the original matrix. No spaces! comma-separated!",
    )
    parser.add_argument(
        "--scalogram",
        help="Use if you want scalogram image to be displayed",
        action="store_true",
    )
    parser.add_argument(
        "--type", help="Provide one of the following: PWD, contact, iPWD"
    )
    parser.add_argument("--pixelSize", help="Provide pixelSize in um")
    parser.add_argument("--cAxis", help="absolute cAxis value for colormap")
    parser.add_argument(
        "--ratio",
        help="Does ratio between matrices. Default: difference",
        action="store_true",
    )
    parser.add_argument(
        "--normalizeMatrix",
        help="Normalizes matrices by maximum. Default: True",
        action="store_true",
    )

    args = parser.parse_args()

    run_parameters = {}

    if args.rootFolder:
        root_folder = args.rootFolder
    else:
        root_folder = "/mnt/PALM_dataserv/DATA/gurgo/Quarantine/Analysis_embryos_cycle_14_16_2020/mixed_embryos_data/26_06_2020_analysis_T=2µm/plotSegments"

    if args.outputFolder:
        output_folder = args.outputFolder
    else:
        output_folder = "none"

    if args.parameters:
        run_parameters["parametersFileName"] = args.parameters
    else:
        run_parameters["parametersFileName"] = "folders_to_load.json"

    if args.label:
        run_parameters["label"] = args.label
    else:
        run_parameters["label"] = "M"

    if args.action:
        run_parameters["action"] = args.action
    else:
        run_parameters["action"] = "all"

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
        run_parameters["axisTicks"] = False

    if args.barcodes:
        run_parameters["barcodes"] = args.barcodes
    else:
        run_parameters["barcodes"] = False

    if args.scalingParameter:
        run_parameters["scalingParameter"] = float(args.scalingParameter)
    else:
        run_parameters["scalingParameter"] = 1.0

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".png"

    if args.shuffle:
        run_parameters["shuffle"] = args.shuffle
    else:
        run_parameters["shuffle"] = 0

    if args.scalogram:
        run_parameters["scalogram"] = args.scalogram
    else:
        run_parameters["scalogram"] = False

    if args.type:
        run_parameters["type"] = args.type
    else:
        run_parameters["type"] = "contact"

    if args.pixelSize:
        run_parameters["pixelSize"] = float(args.pixelSize)
    else:
        run_parameters["pixelSize"] = 1

    if args.cAxis:
        run_parameters["cAxis"] = [float(i) for i in args.cAxis.split(",")]
    else:
        run_parameters["cAxis"] = [0.1]

    if args.ratio:
        run_parameters["ratio"] = args.ratio
    else:
        run_parameters["ratio"] = False

    if args.normalizeMatrix:
        run_parameters["normalizeMatrix"] = args.normalizeMatrix
    else:
        run_parameters["normalizeMatrix"] = False

    run_parameters["outputFolder"] = output_folder
    run_parameters["rootFolder"] = root_folder

    return run_parameters


# =============================================================================
# FUNCTIONS
# =============================================================================


def plotTADs(list_data, run_parameters, data_sets):
    if len(run_parameters["cAxis"]) == 2:
        c_scale = run_parameters["cAxis"][1]
    else:
        c_scale = run_parameters["cAxis"][0]
    print("--------\nClim used: {}\n--------------\n".format(c_scale))

    for idata_set in data_sets:
        if (
            run_parameters["type"] == "contact"
            and "TAD2plot" in list_data[idata_set].keys()
        ):
            Samples = list_data[idata_set]["Folders"]
            TAD2plot = list_data[idata_set]["TAD2plot"]
            segmentLabels = list_data[idata_set]["segmentLabels"]
            segment2plot = list_data[idata_set]["segment2plot"]
            Nplots = len(Samples)

            him_data = AnalysisHiMMatrix(
                run_parameters, os.path.dirname(Samples[segment2plot])
            )
            him_data.load_data()

            m1 = him_data.data["ensembleContactProbability"]
            if run_parameters["normalizeMatrix"]:
                m1 = m1 / m1.max()

            submatrixReference = m1[
                TAD2plot[0] : TAD2plot[1], TAD2plot[0] : TAD2plot[1]
            ]
            fig3 = plt.figure(
                constrained_layout=False,
                figsize=(5 * Nplots, 5),
                dpi=300,
                facecolor="w",
                edgecolor="k",
            )
            nCols, nRows = Nplots, 1
            spec2 = gridspec.GridSpec(ncols=nCols, nrows=nRows, figure=fig3)

            FigList, Yticks, Xticks = [], [], []
            for i_row in range(nRows):
                for i_col in range(nCols):
                    FigList.append(fig3.add_subplot(spec2[i_row, i_col]))
                    if i_row == nRows - 1:
                        Xticks.append(False)
                    else:
                        Xticks.append(False)
                    if i_col == 0:
                        Yticks.append(True)
                    else:
                        Yticks.append(False)

            FigLabels = [isample.split(os.sep)[-2] for isample in Samples]
            legendList = [False] * len(Samples)
            colorbar = [False] * len(Samples)
            i = 0
            for isample, ifigure, i_fig_label, yticks, xticks, legend, icolorbar in zip(
                Samples, FigList, FigLabels, Yticks, Xticks, legendList, colorbar
            ):
                him_data = AnalysisHiMMatrix(run_parameters, os.path.dirname(isample))
                him_data.load_data()

                subMatrix = him_data.data["ensembleContactProbability"][
                    TAD2plot[0] : TAD2plot[1], TAD2plot[0] : TAD2plot[1]
                ]

                if run_parameters["normalizeMatrix"]:
                    subMatrix = subMatrix / subMatrix.max()

                if "ContactProbability_cm" in list_data[idata_set].keys():
                    colormap = list_data[idata_set]["ContactProbability_cm"]

                if run_parameters["ratio"]:
                    subMatrixNormalized = np.log(submatrixReference / subMatrix)
                else:
                    subMatrixNormalized = submatrixReference - subMatrix

                print(
                    "scalingParameters, scale={}, {}".format(
                        run_parameters["scalingParameter"], c_scale
                    )
                )

                n_cells = him_data.n_cells_loaded()

                n_datasets = len(him_data.data["runName"])

                f2_ax1_im = him_data.plot_2d_matrix_simple(
                    ifigure,
                    subMatrixNormalized,
                    list(him_data.data["uniqueBarcodes"]),
                    run_parameters["axisLabel"],
                    run_parameters["axisLabel"],
                    cmtitle=segmentLabels[i],
                    c_min=-c_scale,
                    c_max=c_scale,
                    fontsize=run_parameters["fontsize"],
                    colorbar=icolorbar,
                    axis_ticks=run_parameters["axisTicks"],
                    n_cells=n_cells,
                    n_datasets=n_datasets,
                    show_title=True,
                    fig_title=i_fig_label,
                    c_m=colormap,
                )

                del him_data, subMatrixNormalized
                i += 1
                f2_ax1_im.set_clim(vmin=-c_scale, vmax=c_scale)
                # print("\n\n======--==--=={}\n\n======--==--==".format(c_scale))

            # colorbar=True
            # cbar = fig3.colorbar(f2_ax1_im, ax=ifigure, fraction=0.046, pad=0.04)
            # cbar.minorticks_on()
            # cbar.set_label("difference",fontsize=float(fontsize)*0.85)
            # f2_ax1_im.set_clim(vmin=-c_scale, vmax=c_scale)

            outputFileName2 = run_parameters["outputFileName"].replace(
                "Fig_HiMmatrix", "Fig_TAD"
            )
            print("Output written to {}".format(outputFileName2))
            fig3.savefig(outputFileName2)


def makesplotHiMLineProfile(
    matrixSegmentAnchor,
    unique_barcodes,
    segmentLabels,
    c_scale=0.3,
    c_m="RdBu",
    fontsize=8,
):
    numberSegments = matrixSegmentAnchor.shape[1]

    fig1 = plt.figure(constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f_1 = fig1.add_subplot(spec1[0, 0])  # 16
    pos = f_1.imshow(matrixSegmentAnchor, cmap=c_m)
    f_1.set_ylabel("barcode #", fontsize=1.5 * float(fontsize))
    f_1.set_xlabel("segment ID", fontsize=1.5 * float(fontsize))

    f_1.set_xticks(np.arange(numberSegments))
    f_1.set_xticklabels(segmentLabels, fontsize=fontsize)
    f_1.set_yticks(np.arange(len(unique_barcodes)))
    f_1.set_yticklabels(unique_barcodes, fontsize=fontsize)

    cbar = fig1.colorbar(pos, ax=f_1, fraction=0.046, pad=0.04)
    cbar.minorticks_on()
    # cbar.set_label("difference",fontsize=float(fontsize)*0.85)
    pos.set_clim(vmin=-c_scale, vmax=c_scale)

    return fig1


def plotHiMLineProfile(list_data, run_parameters, data_sets):
    if len(run_parameters["cAxis"]) == 2:
        c_scale = run_parameters["cAxis"][1]
    else:
        c_scale = run_parameters["cAxis"][0]
    print("--------\nClim used: {}\n--------------\n".format(c_scale))
    fontsize = run_parameters["fontsize"]

    for idata_set in data_sets:
        if (
            run_parameters["type"] == "contact"
            and "plotSegment_anchor" in list_data[idata_set].keys()
        ):
            Samples = list_data[idata_set]["Folders"]

            plotSegment_anchor = list_data[idata_set]["plotSegment_anchor"]
            segmentLabels = list_data[idata_set]["segmentLabels"]
            segment2plot = list_data[idata_set]["segment2plot"]
            c_m = list_data[idata_set]["ContactProbability_cm"]

            him_data = AnalysisHiMMatrix(
                run_parameters, os.path.dirname(Samples[segment2plot])
            )
            him_data.load_data()
            # m1=him_data.data["ensembleContactProbability"]
            m1, _ = calculate_contact_probability_matrix(
                him_data.data["SCmatrixCollated"],
                list(him_data.data["uniqueBarcodes"]),
                pixel_size=run_parameters["pixelSize"],
                threshold=0.25,
                norm="nonNANs",
            )

            if run_parameters["normalizeMatrix"]:
                m1 = m1 / m1.max()
            contactsAnchor = m1[plotSegment_anchor, :]

            number_barcodes = him_data.data["ensembleContactProbability"].shape[0]
            numberSegments = len(Samples)

            matrixSegmentAnchor = np.zeros((number_barcodes, numberSegments))

            for iSample, sample in enumerate(Samples):
                him_data = AnalysisHiMMatrix(run_parameters, os.path.dirname(sample))
                him_data.load_data()

                # matrix=him_data.data["ensembleContactProbability"]
                matrix, _ = calculate_contact_probability_matrix(
                    him_data.data["SCmatrixCollated"],
                    list(him_data.data["uniqueBarcodes"]),
                    pixel_size=run_parameters["pixelSize"],
                    threshold=0.25,
                    norm="nonNANs",
                )
                if run_parameters["normalizeMatrix"]:
                    matrix = matrix / matrix.max()

                if run_parameters["ratio"]:
                    matrixSegmentAnchor[:, iSample] = np.log(
                        contactsAnchor / matrix[plotSegment_anchor, :]
                    )
                else:
                    matrixSegmentAnchor[:, iSample] = (
                        contactsAnchor - matrix[plotSegment_anchor, :]
                    )

            unique_barcodes = list(him_data.data["uniqueBarcodes"])

            fig1 = makesplotHiMLineProfile(
                matrixSegmentAnchor,
                unique_barcodes,
                segmentLabels,
                c_scale=c_scale,
                c_m=c_m,
                fontsize=fontsize,
            )
            outputFileName1 = run_parameters["outputFileName"].replace(
                "Fig_HiMmatrix", "Fig_Segment"
            )
            print("Output written to {}".format(outputFileName1))
            fig1.savefig(outputFileName1)

            if "barcodes2plot" in list_data[idata_set].keys():
                barcodes2plot = list_data[idata_set]["barcodes2plot"]
                fig2 = makesplotHiMLineProfile(
                    matrixSegmentAnchor[np.arange(barcodes2plot[0], barcodes2plot[1])],
                    unique_barcodes[barcodes2plot[0] : barcodes2plot[1]],
                    segmentLabels,
                    c_scale=c_scale,
                    c_m=c_m,
                    fontsize=fontsize,
                )
                outputFileName2 = run_parameters["outputFileName"].replace(
                    "Fig_HiMmatrix", "Fig_Segment_subMatrix"
                )
                print("Output written to {}".format(outputFileName2))
                fig2.savefig(outputFileName2)


def plotMultipleHiMmatrices(list_data, run_parameters, data_sets):
    for idata_set in data_sets:
        Samples = list_data[idata_set]["Folders"]

        Nplots = len(Samples)

        fig2 = plt.figure(
            constrained_layout=False,
            figsize=(5 * Nplots, 5),
            dpi=300,
            facecolor="w",
            edgecolor="k",
        )
        # nCols=np.ceil(len(anchors)/2).astype(int)
        nCols = Nplots
        nRows = 1
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

        FigLabels = [isample.split(os.sep)[-2] for isample in Samples]
        legendList = [False] * len(Samples)
        colorbar = [False] * len(Samples)
        # colorbar[-1]=True

        for isample, ifigure, i_fig_label, yticks, xticks, legend, icolorbar in zip(
            Samples, FigList, FigLabels, Yticks, Xticks, legendList, colorbar
        ):
            him_data = AnalysisHiMMatrix(run_parameters, os.path.dirname(isample))
            him_data.load_data()

            if run_parameters["type"] == "contact":
                matrix = him_data.data["ensembleContactProbability"]
                # matrix=normalize_matrix(matrix)
                c_scale = matrix.max() / run_parameters["scalingParameter"]
                if "ContactProbability_cm" in list_data[idata_set].keys():
                    colormap = list_data[idata_set]["ContactProbability_cm"]

            elif run_parameters["type"] == "PWD":
                matrix_sc = him_data.data["SCmatrixCollated"]
                cells_to_plot = list_sc_to_keep(
                    run_parameters, him_data.data["SClabeledCollated"]
                )
                matrix = run_parameters["pixelSize"] * np.nanmedian(
                    matrix_sc[:, :, cells_to_plot], axis=2
                )
                c_scale = 3 * np.nanmedian(matrix) / run_parameters["scalingParameter"]
                if "PWD_cm" in list_data[idata_set].keys():
                    colormap = list_data[idata_set]["PWD_cm"]
                del matrix_sc

            elif run_parameters["type"] == "iPWD":
                matrix_sc = him_data.data["SCmatrixCollated"]
                cells_to_plot = list_sc_to_keep(
                    run_parameters, him_data.data["SClabeledCollated"]
                )
                matrixPWD = run_parameters["pixelSize"] * np.nanmedian(
                    matrix_sc[:, :, cells_to_plot], axis=2
                )
                matrix = np.reciprocal(matrixPWD)
                c_scale = (
                    3
                    * np.reciprocal(np.nanmedian(matrix))
                    / run_parameters["scalingParameter"]
                )
                if "iPWD_cm" in list_data[idata_set].keys():
                    colormap = list_data[idata_set]["iPWD_cm"]
                del matrixPWD, matrix_sc

            print(
                "scalingParameters, scale={}, {}".format(
                    run_parameters["scalingParameter"], c_scale
                )
            )

            n_cells = him_data.n_cells_loaded()

            n_datasets = len(him_data.data["runName"])

            if run_parameters["shuffle"] == 0:
                index = range(matrix.shape[0])
            else:
                index = [int(i) for i in run_parameters["shuffle"].split(",")]
                matrix = shuffle_matrix(matrix, index)

            f2_ax1_im = him_data.plot_2d_matrix_simple(
                ifigure,
                matrix,
                list(him_data.data["uniqueBarcodes"]),
                run_parameters["axisLabel"],
                run_parameters["axisLabel"],
                cmtitle=run_parameters["type"],
                c_min=0,
                c_max=c_scale,
                fontsize=run_parameters["fontsize"],
                colorbar=icolorbar,
                axis_ticks=run_parameters["axisTicks"],
                n_cells=n_cells,
                n_datasets=n_datasets,
                show_title=True,
                fig_title=i_fig_label,
                c_m=colormap,
            )

            del him_data, matrix

        cbar_ax = fig2.add_axes([0.92, 0.20, 0.005, 0.6])
        cbar = fig2.colorbar(f2_ax1_im, cax=cbar_ax, fraction=0.046, pad=0.04)
        ticklabs = cbar.ax.get_yticklabels()
        ticklabs1 = [
            "{:04.2f}".format(i * c_scale / (len(ticklabs) - 1))
            for i in range(len(ticklabs))
        ]
        cbar.ax.set_yticklabels(ticklabs1, fontsize=run_parameters["fontsize"])
        cbar.set_label(
            run_parameters["type"], fontsize=1.2 * float(run_parameters["fontsize"])
        )

        print("Output written to {}".format(run_parameters["outputFileName"]))
        plt.savefig(run_parameters["outputFileName"])
        title_text = "N = {} | n = {}".format(n_cells, n_datasets)
        print("Title: {}".format(title_text))
        print("Output figure: {}".format(run_parameters["outputFileName"]))


# =============================================================================
# MAIN
# =============================================================================


def main():
    print(">>> Producing HiM matrix")
    run_parameters = parse_arguments()

    # loads datasets: parameter files
    filename_list_data_json = (
        run_parameters["rootFolder"] + os.sep + run_parameters["parametersFileName"]
    )
    with open(filename_list_data_json, encoding="utf-8") as json_file:
        list_data = json.load(json_file)

    data_sets = list(list_data.keys())
    if run_parameters["outputFolder"] == "none":
        run_parameters["outputFolder"] = run_parameters["rootFolder"]

    run_parameters["outputFileName"] = (
        run_parameters["outputFolder"]
        + os.sep
        + "Fig_HiMmatrix"
        + "_label:"
        + run_parameters["label"]
        + "_action:"
        + run_parameters["action"]
        + run_parameters["plottingFileExtension"]
    )

    plotMultipleHiMmatrices(list_data, run_parameters, data_sets)

    plotHiMLineProfile(list_data, run_parameters, data_sets)

    plotTADs(list_data, run_parameters, data_sets)

    print("\nDone\n\n")


if __name__ == "__main__":
    main()
