#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:41:05 2020

@author: marcnol

produces movies and structures from single cell PWD matrices

"""

import argparse

# %% imports and plotting settings
import os

import matplotlib

# from mayavi.mlab import *
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

from matrixOperations.HIMmatrixOperations import (
    AnalysisHiMMatrix,
    get_barcodes_per_cell,
    get_coordinates_from_pwd_matrix,
    get_detection_eff_barcodes,
    get_rg_from_pwd,
    kde_fit,
    plot_distance_histograms,
    sort_cells_by_number_pwd,
    write_xyz_2_pdb,
)

font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}

matplotlib.rc("font", **font)

# %% define and loads datasets


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
    parser.add_argument("--nRows", help="The number of cells is set by nRows**2")
    parser.add_argument("--pixelSize", help="Pixel Size in um")
    parser.add_argument("--maxDistance", help="Maximum distance for histograms, in um")

    parser.add_argument(
        "--plottingFileExtension", help="By default: svg. Other options: pdf, png"
    )
    parser.add_argument(
        "--shuffle",
        help="Provide shuffle vector: 0,1,2,3... of the same size or smaller than the original matrix. No spaces! comma-separated!",
    )
    parser.add_argument(
        "--ensembleMatrix",
        help="Use if you want ensembleMatrix to be plotted alongside sc matrices",
        action="store_true",
    )

    parser.add_argument(
        "--plotHistogramMatrix",
        help="Use if you want to plot the PWD histograms for all bin combinations. This is slow!",
        action="store_true",
    )
    parser.add_argument("--minNumberPWD", help="Minimum number of PWD to calculate Rg")
    parser.add_argument(
        "--threshold", help="Maximum accepted PWD to calculate Rg, in px"
    )

    args = parser.parse_args()

    run_parameters = {}

    if args.rootFolder:
        root_folder = args.rootFolder
    else:
        # root_folder = "."
        # root_folder='/home/marcnol/data'+os.sep+'Experiment_18'
        # root_folder = "/mnt/grey/DATA/docPaper_fullDatasets/updatedDatasets/wt_docTAD_nc14"
        root_folder = "/home/marcnol/data/updatedDatasets/wt_docTAD_nc14"

    if args.outputFolder:
        output_folder = args.outputFolder
    else:
        output_folder = root_folder + os.sep + "figureSingleCell"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        print("Folder created: {}".format(output_folder))

    if args.parameters:
        run_parameters["parametersFileName"] = args.parameters
    else:
        run_parameters["parametersFileName"] = "folders_to_load.json"

    if args.label:
        run_parameters["label"] = args.label
    else:
        run_parameters["label"] = "doc"

    if args.action:
        run_parameters["action"] = args.action
    else:
        run_parameters["action"] = "all"

    if args.fontsize:
        run_parameters["fontsize"] = args.fontsize
    else:
        run_parameters["fontsize"] = 12

    if args.pixelSize:
        run_parameters["pixelSize"] = float(args.pixelSize)
    else:
        run_parameters["pixelSize"] = 0.1

    if args.maxDistance:
        run_parameters["maxDistance"] = float(args.maxDistance)
    else:
        run_parameters["maxDistance"] = 4.0

    if args.threshold:
        run_parameters["threshold"] = float(args.threshold)
    else:
        run_parameters["threshold"] = 8

    if args.minNumberPWD:
        run_parameters["minNumberPWD"] = args.minNumberPWD
    else:
        run_parameters["minNumberPWD"] = 6

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

    if args.plottingFileExtension:
        run_parameters["plottingFileExtension"] = "." + args.plottingFileExtension
    else:
        run_parameters["plottingFileExtension"] = ".png"

    if args.shuffle:
        run_parameters["shuffle"] = args.shuffle
    else:
        run_parameters["shuffle"] = 0

    if args.nRows:
        run_parameters["nRows"] = int(args.nRows)
    else:
        run_parameters["nRows"] = int(10)

    if args.ensembleMatrix:
        run_parameters["ensembleMatrix"] = args.ensembleMatrix
    else:
        run_parameters["ensembleMatrix"] = False

    if args.plotHistogramMatrix:
        run_parameters["plotHistogramMatrix"] = args.plotHistogramMatrix
    else:
        run_parameters["plotHistogramMatrix"] = False

    return root_folder, output_folder, run_parameters


def returnCellsHighestNumberPWD(sorted_values, n):
    cell_id = []
    Npwd = []

    for i in range(1, n + 1):
        cell_id.append(sorted_values[-i][0])
        Npwd.append(sorted_values[-i][1])

    return cell_id, Npwd


def visualize3D(coordinates, colors=[], cmap="hsv", title=[], output="visualize3D.png"):
    fig = plt.figure()
    fig.set_size_inches((10, 10))

    ax = plt.axes(projection="3d")

    xdata, ydata, zdata, barcode_id = [], [], [], []
    for i, r in enumerate(coordinates):
        xdata.append(r[0])
        ydata.append(r[1])
        zdata.append(r[2])
        barcode_id.append(i)

    if len(colors) == 0:
        colors = np.arange(coordinates.shape[0])
        colors = colors.flatten()

    ax.plot3D(xdata, ydata, zdata, "black")
    # 'terrain_r'
    ax.scatter3D(
        xdata,
        ydata,
        zdata,
        c=colors,
        s=200,
        cmap=cmap,
        marker="o",
        edgecolors="k",
        linewidths=2,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.set_zticklabels(())
    fig.savefig(output)
    plt.close(fig)


def visualize2D(
    coordinateList, colors=[], cmap="hsv", titles=[], output="visualize2D.png"
):
    nRows = len(coordinateList)

    fig, allAxes = plt.subplots(1, nRows)
    fig.set_size_inches((10 * nRows, 10))
    axes = allAxes.ravel()

    for coordinates, ax, title in zip(coordinateList, axes, titles):
        if len(colors) == 0:
            colors = np.arange(coordinates.shape[0])
            colors = colors.flatten()

        xdata, ydata, zdata, barcode_id = [], [], [], []
        for i, r in enumerate(coordinates):
            xdata.append(r[0])
            ydata.append(r[1])
            zdata.append(r[2])
            barcode_id.append(i)

        ax.plot(xdata, ydata, "black")
        # 'terrain_r'
        ax.scatter(
            xdata,
            ydata,
            s=200,
            c=colors,
            cmap=cmap,
            marker="o",
            edgecolors="k",
            linewidths=2,
            alpha=0.7,
            vmax=1,
            vmin=0,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(title)

        fig.savefig(output)
        plt.close(fig)


def plotTrajectories(
    him_data, run_parameters, outputFileNameRoot, cell_id, sc_matrix, mode="matplotlib"
):
    pwd_matrix = sc_matrix[:, :, cell_id]
    EnsembleMatrix = 1 / him_data.data["ensembleContactProbability"]

    ATACseqMatrix = (
        np.array(him_data.list_data[him_data.dataset_name]["BarcodeColormap"]) / 10
    )
    colors = np.atleast_2d(ATACseqMatrix).flatten()

    singleCellTitle = "Cell #" + str(cell_id)
    ensembleTitle = "Ensemble"

    # removes nans
    for i in range(pwd_matrix.shape[0]):
        pwd_matrix[i, i] = 0
        EnsembleMatrix[i, i] = 0

    # gets coordinates and saves in PDB format
    EnsembleMatrix[np.isnan(EnsembleMatrix)] = 0  # removes NaNs from matrix
    coordinatesEnsemble = get_coordinates_from_pwd_matrix(EnsembleMatrix)

    pwd_matrix[np.isnan(pwd_matrix)] = 0  # removes NaNs from matrix
    coordinates = run_parameters["pixelSize"] * get_coordinates_from_pwd_matrix(
        pwd_matrix
    )
    output_filename_pdb = (
        outputFileNameRoot + "_SingleCellTrajectory:" + str(cell_id) + ".pdb"
    )
    write_xyz_2_pdb(output_filename_pdb, coordinates)

    # makes plots
    cmap = "tab10"
    output = (
        outputFileNameRoot
        + "_2DsingleCell:"
        + str(cell_id)
        + run_parameters["plottingFileExtension"]
    )
    visualize2D(
        [coordinates, coordinatesEnsemble],
        colors=colors,
        cmap=cmap,
        titles=[singleCellTitle, ensembleTitle],
        output=output,
    )

    output = (
        outputFileNameRoot
        + "_3DensembleMatrix"
        + run_parameters["plottingFileExtension"]
    )
    visualize3D(
        coordinatesEnsemble,
        colors=colors,
        cmap=cmap,
        title=ensembleTitle,
        output=output,
    )

    output = (
        outputFileNameRoot
        + "_3DsingleCell:"
        + str(cell_id)
        + run_parameters["plottingFileExtension"]
    )

    if mode == "matplotlib":
        visualize3D(
            coordinates, colors=colors, cmap=cmap, title=singleCellTitle, output=output
        )

    # else:
    #     visualize3D_mayavi(coordinates,colors=colors,cmap='rainbow',title=singleCellTitle, output=output)


def plot_sc_matrix(
    him_data,
    cell_id,
    run_parameters,
    outputFileNameRoot="sc_matrix.png",
    ensembleMatrix=False,
    searchPattern="_scMatrix:",
):
    dataset_name = list(him_data.list_data.keys())[0]

    vmax = him_data.list_data[dataset_name]["iPWD_clim"]
    cmap = him_data.list_data[dataset_name]["iPWD_cm"]
    sc_matrix = him_data.sc_matrix_selected
    singleCellTitle = "Cell #" + str(cell_id)

    if ensembleMatrix:
        ensembleTitle = "Ensemble"
        EnsembleMatrix = 1 / him_data.data["ensembleContactProbability"]

    pwd_matrix = sc_matrix[:, :, cell_id]

    if ensembleMatrix:
        fig, allAxes = plt.subplots(1, 2)
        ax = allAxes.ravel()
        fig.set_size_inches((20, 10))
    else:
        fig, allAxes = plt.subplots(1, 1)
        fig.set_size_inches((10, 10))
        ax = [allAxes]

    unique_barcodes = 1 + np.arange(pwd_matrix.shape[0])

    p_1 = ax[0].imshow(1 / pwd_matrix, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(p_1, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title(singleCellTitle)
    plt.xticks(np.arange(pwd_matrix.shape[0]), unique_barcodes)
    plt.yticks(np.arange(pwd_matrix.shape[0]), unique_barcodes)

    if ensembleMatrix:
        p_2 = ax[1].imshow(1 / EnsembleMatrix[:, :], cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(p_2, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_title(ensembleTitle)

    output = (
        outputFileNameRoot
        + searchPattern
        + str(cell_id)
        + run_parameters["plottingFileExtension"]
    )
    plt.savefig(output)
    plt.close(fig)


def plotsSubplot_sc_matrices(him_data, nRows, output="subplotMatrices.png"):
    dataset_name = list(him_data.list_data.keys())[0]

    sc_matrix, sorted_values, n_cells = sort_cells_by_number_pwd(him_data)

    # displays plots
    Ncells2Process = nRows**2
    cell_id, Npwd = returnCellsHighestNumberPWD(sorted_values, Ncells2Process)

    fig, allAxes = plt.subplots(nRows, nRows)
    fig.set_size_inches((50, 50))
    ax = allAxes.ravel()

    cmap = him_data.list_data[dataset_name]["ContactProbability_cm"]
    vmax = him_data.list_data[dataset_name]["iPWD_clim"]

    iplot = 0
    for i_cell in cell_id:
        ax[iplot].imshow(1 / sc_matrix[:, :, i_cell], cmap=cmap, vmin=0, vmax=vmax)
        ax[iplot].set_xticklabels(())
        ax[iplot].set_yticklabels(())
        ax[iplot].set_axis_off()
        ax[iplot].set_title(str(i_cell))

        iplot += 1

    plt.savefig(output)
    plt.close(fig)
    return cell_id, sc_matrix


def plotsBarcodesPerCell(sc_matrix, run_parameters, outputFileNameRoot="./"):
    num_barcodes = get_barcodes_per_cell(sc_matrix)
    maxNumberBarcodes = sc_matrix.shape[0]

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    ax.hist(num_barcodes, bins=range(2, maxNumberBarcodes + 1))
    ax.set_xlabel("number of barcodes")
    ax.set_ylabel("counts")

    output = (
        outputFileNameRoot
        + "_SChistBarcodesPerCell"
        + run_parameters["plottingFileExtension"]
    )
    plt.savefig(output)
    plt.close(fig)


def plotsBarcodesEfficiencies(
    sc_matrix, run_parameters, unique_barcodes, outputFileNameRoot="./"
):
    eff = get_detection_eff_barcodes(sc_matrix)

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    ax.bar(unique_barcodes, eff)
    ax.set_xlabel("barcode ID")
    ax.set_ylabel("efficiency")
    ax.set_xticks(np.arange(len(eff)))
    ax.set_xticklabels(unique_barcodes)

    output = (
        outputFileNameRoot
        + "_SCBarcodesEfficiency"
        + run_parameters["plottingFileExtension"]
    )
    plt.savefig(output)
    plt.close(fig)


def plotsRgvalues(
    him_data,
    nRows,
    run_parameters,
    output_filename="./RgValues.png",
    min_number_pwd=6,
    threshold=6,
    bandwidths=10 ** np.linspace(-1.5, 0, 20),
):
    print("Threshold = {} px | min number PWDs = {}".format(threshold, min_number_pwd))

    sc_matrix, sorted_values, n_cells = sort_cells_by_number_pwd(him_data)
    Ncells2Process = nRows**2
    selectedCellsIDs, Npwd = returnCellsHighestNumberPWD(sorted_values, Ncells2Process)

    # calculates Rg for all cells
    RgList = []
    for cell_id in selectedCellsIDs:
        RgList.append(
            run_parameters["pixelSize"]
            * get_rg_from_pwd(
                sc_matrix[:, :, cell_id],
                min_number_pwd=min_number_pwd,
                threshold=threshold,
            )
        )

    RgListArray = np.array(RgList)
    maxRange = 1.5 * RgListArray.max()

    # plots single Rg as bars
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
        subplot_kw={"xlim": (0, maxRange), "ylim": (-0.02, 1.2)},
    )
    ax.plot(RgListArray, np.full_like(RgListArray, -0.01), "|k", markeredgewidth=0.5)

    # calculates and displays median
    mean = np.nanmedian(RgListArray)
    print("Median Rg = {}".format(mean))
    ax.set_ylabel("counts")
    ax.set_xlabel("Rg, " + r"$\mu$m")
    x_d = np.linspace(0, maxRange, 100)

    # KDE fit
    # fist finds best bandwidth
    print("Calculating optimal KDE bandwidth...")

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=LeaveOneOut()
    )
    grid.fit(RgListArray[:, None])
    bandwidth = grid.best_params_["bandwidth"]
    print("bandwidth = {}".format(bandwidth))

    # calculates KDE with optimal bandwidth
    logprob, kde = kde_fit(RgListArray, x_d, bandwidth=bandwidth)
    maxlogprob = logprob.max()
    ax.fill_between(x_d, np.exp(logprob) / np.exp(maxlogprob), alpha=0.3)

    mean = x_d[np.argmax(logprob, axis=0)]
    print("KDE max Rg = {}".format(mean))

    ax.axvline(x=mean, color="black", linestyle=(0, (5, 5)))

    plt.savefig(output_filename)
    plt.close(fig)
    return RgList


def makesPlotHistograms(
    him_data,
    run_parameters,
    output_filename="./HiMhistograms.png",
    mode="KDE",
    kernel_width=0.25,
    optimize_kernel_width=False,
):
    sc_matrix, sorted_values, n_cells = sort_cells_by_number_pwd(him_data)

    plot_distance_histograms(
        sc_matrix,
        run_parameters["pixelSize"],
        output_filename,
        mode="KDE",
        kernel_width=0.25,
        optimize_kernel_width=False,
        max_distance=run_parameters["maxDistance"],
    )


# %%
# =============================================================================
# MAIN
# =============================================================================


def main():
    print(">>> Producing HiM matrix")
    root_folder, output_folder, run_parameters = parse_arguments()
    him_data = AnalysisHiMMatrix(run_parameters, root_folder)
    him_data.load_data()
    him_data.retrieve_sc_matrix()
    if output_folder == "none":
        output_folder = him_data.data_folder
    outputFileNameRoot = (
        output_folder
        + os.sep
        + "Fig_SCmatrices"
        + "_dataset1:"
        + him_data.dataset_name
        + "_label:"
        + run_parameters["label"]
        + "_action:"
        + run_parameters["action"]
    )
    dataset_name = list(him_data.list_data.keys())[0]
    print("Data output: {}".format(outputFileNameRoot))

    # "makes subplots with sc 1/PWD matrices"
    print("\n>>>Plotting subplots with 1/PWD matrices<<<\n")
    nRows = run_parameters["nRows"]
    output = (
        outputFileNameRoot + "_scMatrices" + run_parameters["plottingFileExtension"]
    )
    cellID_most_PWDs, sc_matrix = plotsSubplot_sc_matrices(
        him_data, nRows, output=output
    )

    # "calculates the number of barcodes per cell and makes histograms"
    print("\n>>>Calculating distribution of barcodes<<<\n")
    plotsBarcodesPerCell(
        sc_matrix, run_parameters, outputFileNameRoot=outputFileNameRoot
    )

    # "calculates the detection efficiency for each barcode"
    print("\n>>>Calculating detection efficiency distribution<<<\n")
    plotsBarcodesEfficiencies(
        sc_matrix,
        run_parameters,
        list(him_data.data["uniqueBarcodes"]),
        outputFileNameRoot=outputFileNameRoot,
    )

    # "calculates the Rg for each cell from the PWD sc matrix"
    print("\n>>>Calculating Rg distributions<<<\n")
    output = outputFileNameRoot + "_RgValues" + run_parameters["plottingFileExtension"]
    plotsRgvalues(
        him_data,
        nRows,
        run_parameters,
        output_filename=output,
        min_number_pwd=int(run_parameters["minNumberPWD"]),
        threshold=float(run_parameters["threshold"]),
        bandwidths=10 ** np.linspace(-1, 0, 20),
    )

    # plots distance histograms
    if run_parameters["plotHistogramMatrix"]:
        print("\n>>>Plotting distance histograms<<<\n")
        output = (
            outputFileNameRoot
            + "_HistogramPWDs"
            + run_parameters["plottingFileExtension"]
        )
        makesPlotHistograms(
            him_data,
            run_parameters,
            output_filename=output,
            mode="KDE",
            kernel_width=0.25,
            optimize_kernel_width=False,
        )

    # "plots trajectories for selected cells"
    print("\n>>>Plotting trajectories for selected cells<<<\n")
    if "CellIDs" in him_data.list_data[dataset_name].keys():
        CellIDs = him_data.list_data[dataset_name]["CellIDs"]
        print("CellIDs to process: {}".format(CellIDs))
        for cell_id in CellIDs:
            if cell_id < him_data.sc_matrix_selected.shape[2]:
                #  Plots sc 1/PWD matrix and ensemble 1/PWD matrix together
                plot_sc_matrix(
                    him_data,
                    cell_id,
                    run_parameters,
                    outputFileNameRoot,
                    ensembleMatrix=run_parameters["ensembleMatrix"],
                )

                # plots trajectories
                plotTrajectories(
                    him_data, run_parameters, outputFileNameRoot, cell_id, sc_matrix
                )

    print("\nDone\n\n")


if __name__ == "__main__":
    main()
