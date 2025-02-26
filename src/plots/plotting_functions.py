#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:34:59 2023

@author: marcnol
"""
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap, ranksums
from tqdm import trange

from matrixOperations.HIMmatrixOperations import (
    calculate_contact_probability_matrix,
    shuffle_matrix,
)

rng = np.random.default_rng()


def bootstrapping(x, N_bootstrap=9999):
    data = (x,)  # samples must be in a sequence

    res = bootstrap(
        data,
        np.mean,
        confidence_level=0.9,
        n_resamples=N_bootstrap,
        batch=None,
        random_state=rng,
    )

    return res.bootstrap_distribution


def bootstraps_matrix(m, N_bootstrap=9999):
    n_bins = m.shape[0]
    mean_bs = np.zeros((n_bins, n_bins))
    mean_error = np.zeros((n_bins, n_bins))
    print(f"$ n bins: {n_bins}")

    for i in trange(n_bins):
        for j in range(i + 1, n_bins):
            if i != j:
                # gets distribution and removes nans
                x = m[i, j, :]
                x = x[~np.isnan(x)]

                # bootstraps distribution
                bs = bootstrapping(x, N_bootstrap=N_bootstrap)

                # gets mean and std of mean
                mean_bs[i, j] = np.median(bs)
                mean_error[i, j] = np.std(bs)

                # symmetrizes matrix
                mean_bs[j, i] = mean_bs[i, j]
                mean_error[j, i] = np.std(bs)

    for i in range(n_bins):
        mean_bs[i, i], mean_error[i, i] = np.nan, np.nan

    return mean_bs, mean_error


def gets_matrix(run_parameters, scPWDMatrix_filename="", uniqueBarcodes=""):

    if os.path.exists(scPWDMatrix_filename):
        sc_matrix = np.load(scPWDMatrix_filename)
        print(f"$ Loaded: {scPWDMatrix_filename}")
    else:
        print("*** Error: could not find {}".format(scPWDMatrix_filename))
        sys.exit(-1)

    n_cells = sc_matrix.shape[2]

    cells2Plot = range(n_cells)

    print("$ N traces to plot: {}/{}".format(len(cells2Plot), sc_matrix.shape[2]))

    uniqueBarcodes = list(np.loadtxt(uniqueBarcodes, delimiter=" "))
    uniqueBarcodes = [int(x) for x in uniqueBarcodes]
    print(f"$ unique barcodes loaded: {uniqueBarcodes}")

    print(f"$ averaging method: {run_parameters['dist_calc_mode']}")

    if run_parameters["cMax"] == 0:
        cScale = (
            sc_matrix[~np.isnan(sc_matrix)].max() / run_parameters["scalingParameter"]
        )
    else:
        cScale = run_parameters["cMax"]

    print(
        "$ loaded cScale: {} | used cScale: {}".format(
            run_parameters["scalingParameter"], cScale
        )
    )

    outputFileName = (
        run_parameters["outputFolder"]
        + os.sep
        + "Fig_"
        + os.path.basename(scPWDMatrix_filename).split(".")[0]
    )

    if run_parameters["shuffle"] == 0:
        index = range(sc_matrix.shape[0])
    else:
        index = [int(i) for i in run_parameters["shuffle"].split(",")]
        sc_matrix = shuffle_matrix(sc_matrix, index)

    if run_parameters["dist_calc_mode"] == "proximity":
        # calculates and plots contact probability matrix from merged samples/datasets
        print("$ calculating proximity matrix")
        sc_matrix, n_cells = calculate_contact_probability_matrix(
            sc_matrix,
            uniqueBarcodes,
            run_parameters["pixelSize"],
            threshold=run_parameters["proximity_threshold"],
            norm=run_parameters["matrix_norm_mode"],
        )

    fileNameEnding = (
        "_"
        + run_parameters["dist_calc_mode"]
        + "_"
        + run_parameters["matrix_norm_mode"]
        + "_"
        + str(run_parameters["cMax"])
    )

    return sc_matrix, uniqueBarcodes, cScale, n_cells, outputFileName, fileNameEnding


def Wilcoxon_matrix(
    m1,
    m2,
    uniqueBarcodes,
):

    nbins = len(uniqueBarcodes)
    result = np.zeros((nbins, nbins))

    for i in range(nbins):
        for j in range(nbins):
            if i != j:
                x, y = m1[i, j, :], m2[i, j, :]
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                a, p = ranksums(x, y)  # , nan_policy='omit')
                result[i, j] = p
    return result


def plot_Wilcoxon_matrix(
    m1,
    m2,
    uniqueBarcodes,
    normalize=1,
    ratio=True,
    c_scale=0.6,
    axisLabel=True,
    fontsize=22,
    axis_ticks=False,
    outputFileName="",
    fig_title="",
    plottingFileExtension=".png",
    n_cells=0,
    cmap="RdBu",
):

    cmtitle = "log10(p-value)"
    fig_title = "Wilcoxon's rank sum test"
    print(f"$ normalization factor: {normalize}")

    m2 = m2 / normalize

    print(f"$ max_m1 = {np.nanmax(m1)} \t max_m2 = {np.nanmax(m2)}")

    result = Wilcoxon_matrix(
        m1,
        m2,
        uniqueBarcodes,
    )

    fig1 = plt.figure(constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f_1 = fig1.add_subplot(spec1[0, 0])  # 16

    result = np.log10(result)

    plot_2d_matrix_simple(
        f_1,
        result,
        uniqueBarcodes,
        yticks=axisLabel,
        xticks=axisLabel,
        cmtitle=cmtitle,
        fig_title=fig_title,
        c_min=-3,
        c_max=0,  # log10(0.05) = -1.3
        fontsize=fontsize,
        colorbar=True,
        axis_ticks=axis_ticks,
        c_m=cmap,
        show_title=True,
        n_cells=n_cells,
        n_datasets=2,
    )
    plt.savefig(outputFileName + "_Wilcoxon" + plottingFileExtension)
    print(
        "$ Output figure: {}".format(
            outputFileName + "_Wilcoxon" + plottingFileExtension
        )
    )
    np.save(outputFileName + "_Wilcoxon" + ".npy", result)
    print("$ Output data: {}".format(outputFileName + "_Wilcoxon" + ".npy"))


def normalize_matrix(m1, m2, mode="none"):
    print("$ Normalization: {}".format(mode))

    if "maximum" in mode:  # normalizes by maximum
        m1_norm = 1.0  # np.nanmax(m1)
        m2_norm = np.nanmax(m2) / np.nanmax(m1)
    elif len(mode.split(",")) > 1:  # normalizes by bin
        N = mode.split(",")
        m1_norm = 1
        m2_norm = m2[int(N[0]), int(N[1])] / m1[int(N[0]), int(N[1])]
    elif "none" in mode:  # no normalization
        m1_norm = 1
        m2_norm = 1
    else:  # normalizes by given factor
        norm_factor = float(mode)
        m1_norm = 1
        m2_norm = norm_factor

    print("Normalizations: m1= {} | m2={}".format(m1_norm, m2_norm))

    m1 = m1 / m1_norm
    m2 = m2 / m2_norm

    return m1, m2, m2_norm


def plot_2d_matrix_simple(
    ifigure,
    matrix,
    unique_barcodes,
    yticks,
    xticks,
    cmtitle="probability",
    c_min=0,
    c_max=1,
    c_m="coolwarm",
    fontsize=12,
    colorbar=False,
    axis_ticks=False,
    n_cells=0,
    n_datasets=0,
    show_title=True,
    fig_title="",
):
    pos = ifigure.imshow(matrix, cmap=c_m)  # colormaps RdBu seismic

    if show_title:
        title_text = f"{fig_title} | N = {n_cells} | n = {n_datasets}"
        ifigure.title.set_text(title_text)

    # plots figure
    if xticks:
        ifigure.set_xlabel("barcode #", fontsize=fontsize)
        if not axis_ticks:
            ifigure.set_xticklabels(())
        else:

            print(f"barcodes:{unique_barcodes}")
            ifigure.set_xticks(np.arange(matrix.shape[0]), unique_barcodes)
            # ifigure.set_xticklabels(unique_barcodes)

    else:
        ifigure.set_xticklabels(())

    if yticks:
        ifigure.set_ylabel("barcode #", fontsize=fontsize)
        if not axis_ticks:
            ifigure.set_yticklabels(())
        else:
            ifigure.set_yticks(np.arange(matrix.shape[0]), unique_barcodes)
            # ifigure.set_yticklabels(unique_barcodes)
    else:
        ifigure.set_yticklabels(())

    for xtick, ytick in zip(
        ifigure.xaxis.get_majorticklabels(), ifigure.yaxis.get_majorticklabels()
    ):
        xtick.set_fontsize(fontsize)
        ytick.set_fontsize(fontsize)

    # plt.xticks(
    #     np.arange(matrix.shape[0]), unique_barcodes, fontsize=fontsize
    # )
    # plt.yticks(
    #     np.arange(matrix.shape[0]), unique_barcodes, fontsize=fontsize
    # )
    if colorbar:
        cbar = plt.colorbar(pos, ax=ifigure, fraction=0.046, pad=0.04)
        cbar.minorticks_on()
        cbar.set_label(cmtitle, fontsize=float(fontsize) * 0.85)
        pos.set_clim(vmin=c_min, vmax=c_max)

    pos.set_clim(vmin=c_min, vmax=c_max)

    return pos


def plot_matrix_difference(
    m1,
    m2,
    uniqueBarcodes,
    normalize="none",
    ratio=True,
    c_scale=0.6,
    axisLabel=True,
    fontsize=22,
    axis_ticks=False,
    outputFileName="",
    fig_title="",
    plottingFileExtension=".png",
    n_cells=0,
    cmap="RdBu",
):

    # plots difference/ratio matrix
    fig1 = plt.figure(constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f_1 = fig1.add_subplot(spec1[0, 0])  # 16

    if "none" in normalize:  # sets default operation
        mode = "maximum"
    else:
        mode = normalize

    _m1, _m2 = m1.copy(), m2.copy()

    _m1, _m2, _ = normalize_matrix(_m1, _m2, mode)

    print(f"$ max_m1 = {np.nanmax(_m1)} \t max_m2 = {np.nanmax(_m2)}")
    if ratio:
        matrix = np.log2(_m1 / _m2)
        cmtitle = "log(ratio)"
        fig_title = "log2(Dataset1/Dataset2)"
        print("$ calculating ratio")
    else:
        matrix = _m1 - _m2
        cmtitle = "difference"
        fig_title = "Dataset1-Dataset2"
        print("$ calculating difference")

    print("$ Clim used: {}\n".format(c_scale))

    plot_2d_matrix_simple(
        f_1,
        matrix,
        uniqueBarcodes,
        yticks=axisLabel,
        xticks=axisLabel,
        cmtitle=cmtitle,
        fig_title=fig_title,
        c_min=-c_scale,
        c_max=c_scale,
        fontsize=fontsize,
        colorbar=True,
        axis_ticks=axis_ticks,
        c_m=cmap,
        show_title=True,
        n_cells=n_cells,
        n_datasets=2,
    )
    plt.savefig(outputFileName + "_difference" + plottingFileExtension)
    print("Output figure: {}".format(outputFileName))


def plot_mixed_matrix(
    m1,
    m2,
    uniqueBarcodes,
    normalize="none",
    axisLabel=False,
    fontsize=22,
    axis_ticks=False,
    cAxis=0.6,
    outputFileName="",
    fig_title="",
    plottingFileExtension=".png",
    n_cells=0,
    cmap="RdBu",
):

    # plots mixed matrix
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig2)
    f2 = fig2.add_subplot(spec2[0, 0])  # 16

    _m1, _m2 = m1.copy(), m2.copy()

    if "none" in normalize:  # sets default operation
        mode = "none"
    else:
        mode = normalize

    _m1, _m2, _ = normalize_matrix(_m1, _m2, mode)
    print(f"$ max_m1 = {np.nanmax(_m1)} \t max_m2 = {np.nanmax(_m2)}")

    matrix2 = _m1

    fig_title = "mixed matrix"

    for i in range(matrix2.shape[0]):
        for j in range(0, i):
            matrix2[i, j] = _m2[i, j]

    plot_2d_matrix_simple(
        f2,
        matrix2,
        list(uniqueBarcodes),
        yticks=axisLabel,
        xticks=axisLabel,
        cmtitle=fig_title,
        fig_title=fig_title,
        show_title=True,
        c_min=0,
        c_max=cAxis,
        fontsize=fontsize,
        colorbar=True,
        axis_ticks=axis_ticks,
        c_m=cmap,
        n_cells=n_cells,
        n_datasets=2,
    )
    plt.savefig(outputFileName + "_mixed_matrix" + plottingFileExtension)
    np.save(outputFileName + "_mixed_matrix.npy", matrix2)
    print(
        "Output figure: {}".format(
            outputFileName + "_mixed_matrix" + plottingFileExtension
        )
    )
