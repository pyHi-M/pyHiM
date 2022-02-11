#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 08:40:30 2022

@author: marcnol

This script:
    - iterates over chromatin traces
        - calculates the pair-wise distances for each single-cell mask
        - outputs are:
            - Table with #cell #PWD #coordinates (e.g. buildsPWDmatrix_3D_order:0_ROI:1.ecsv)
            - NPY array with single cell PWD single cell matrices (e.g. buildsPWDmatrix_3D_HiMscMatrix.npy)
            - NPY array with barcode identities (e.g. buildsPWDmatrix_3D_uniqueBarcodes.ecsv)
            - the files with no "3D" tag contain data analyzed using 2D localizations.

    - Single-cell results are combined together to calculate:
        - Distribution of pairwise distance for each barcode combination
        - Ensemble mean pairwise distance matrix using mean of distribution
        - Ensemble mean pairwise distance matrix using Kernel density estimation
        - Ensemble Hi-M matrix using a predefined threshold
        - For each of these files, there is an image in PNG format saved. Images containing "3D" are for 3D other are for 2D.


"""

# =============================================================================
# IMPORTS
# =============================================================================

import glob, os, sys
import uuid
import re
import numpy as np
from tqdm.contrib import tzip
from tqdm import trange
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

from astropy.table import Table, unique

from photutils.segmentation import SegmentationImage

from fileProcessing.fileManagement import (
    folders,
    writeString2File,
    printLog,
    getDictionaryValue,
)

from matrixOperations.HIMmatrixOperations import plotMatrix, plotDistanceHistograms, calculateContactProbabilityMatrix
from matrixOperations.build_traces import initialize_module
from matrixOperations.chromatin_trace_table import chromatin_trace_table


# to remove in a future version
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CLASSES
# =============================================================================


class build_matrix:
    def __init__(self, param):

        self.param = param

        self.initialize_parameters()

        # initialize with default values
        self.currentFolder = []

    def initialize_parameters(self):
        # initializes parameters from param

        self.tracing_method = getDictionaryValue(self.param.param["buildsPWDmatrix"], "tracing_method",  default="masking")
        self.zBinning = getDictionaryValue(self.param.param["acquisition"], "zBinning", default=1)
        self.pixelSizeXY = getDictionaryValue(self.param.param["acquisition"], "pixelSizeXY", default=0.1)
        self.pixelSizeZ_0 = getDictionaryValue(self.param.param["acquisition"], "pixelSizeZ", default=0.25)
        self.pixelSizeZ = self.zBinning * self.pixelSizeZ_0
        self.pixelSize = [self.pixelSizeXY, self.pixelSizeXY, self.pixelSizeZ]
        self.availableMasks = getDictionaryValue(self.param.param["buildsPWDmatrix"], "masks2process",  default={"nuclei":"DAPI"})
        self.logNameMD = self.param.param["fileNameMD"]
        self.mask_expansion = getDictionaryValue(self.param.param["buildsPWDmatrix"], "mask_expansion", default=8)
        self.availableMasks = self.param.param["buildsPWDmatrix"]["masks2process"]

    def calculatesPWDsingleMask(self, r1, r2):
        """
        Calculates PWD between barcodes detected in a given mask. For this:
            - converts xyz pixel coordinates into nm using self.pixelSize dictionary
            - calculates pair-wise distance matrix in nm
            - converts it into pixel units using self.pixelSize['x'] as an isotropic pixelsize.

        Parameters
        ----------
        r1: list of floats with xyz coordinates for spot 1 in microns
        r2: list of floats with xyz coordinates for spot 2 in microns

        Returns
        -------
        Returns pairwise distance matrix between barcodes in microns

        """

        x = np.array([r1[0], r2[0]])
        y = np.array([r1[1], r2[1]])
        z = np.array([r1[2], r2[2]])

        R_mum = np.column_stack((x, y, z))

        P = pairwise_distances(R_mum)

        return P

    def buildsdistanceMatrix(self, mode="mean"):
        """
        Builds pairwise distance matrix from a coordinates table

        Parameters
        ----------
        mode : string, optional
            The default is "mean": calculates the mean distance if there are several combinations possible.
            "min": calculates the minimum distance if there are several combinations possible.
            "last": keeps the last distance calculated

        Returns
        -------
        self.SCmatrix the single-cell PWD matrix
        self.meanSCmatrix the ensamble PWD matrix (mean of SCmatrix without nans)
        self.uniqueBarcodes list of unique barcodes

        """
        # detects number of unique traces from trace table
        numberMatrices = len(unique(self.trace_table.data,keys='Trace_ID'))

        # finds unique barcodes from trace table
        uniqueBarcodes = unique(self.trace_table.data,keys='Barcode #')['Barcode #'].data
        numberUniqueBarcodes = uniqueBarcodes.shape[0]

        printLog(f"$ Found {numberUniqueBarcodes} barcodes and {numberMatrices} traces.","INFO")

        # Initializes SCmatrix
        SCmatrix = np.zeros((numberUniqueBarcodes, numberUniqueBarcodes, numberMatrices))
        SCmatrix[:] = np.NaN

        # loops over traces
        coord_labels = ['x','y','z']

        data_traces = self.trace_table.data.group_by("Trace_ID")
        for trace, trace_id, itrace in zip(data_traces.groups, data_traces.groups.keys, range(numberMatrices)):

            barcodes2Process = trace["Barcode #"].data

            # loops over barcodes detected in cell mask: barcode1
            for barcode1, ibarcode1 in zip(barcodes2Process, range(len(barcodes2Process))):
                indexBarcode1 = np.nonzero(uniqueBarcodes == barcode1)[0][0]

                # gets coordinates for barcode 1
                r1 = [trace[coord_label][ibarcode1].data for coord_label in coord_labels]

                # loops over barcodes detected in cell mask: barcode2
                for barcode2, ibarcode2 in zip(barcodes2Process, range(len(barcodes2Process))):
                    indexBarcode2 = np.nonzero(uniqueBarcodes == barcode2)[0][0]

                    # gets coordinates for barcode 2
                    r2 = [trace[coord_label][ibarcode2].data for coord_label in coord_labels]

                    if barcode1 != barcode2:

                        # attributes distance from the PWDmatrix field in the scPWDitem table
                        newdistance = self.calculatesPWDsingleMask(r1, r2)

                        # inserts value into SCmatrix
                        if mode == "last":
                            SCmatrix[indexBarcode1][indexBarcode2][itrace] = newdistance
                        elif mode == "mean":
                            SCmatrix[indexBarcode1][indexBarcode2][itrace] = np.nanmean(
                                [newdistance, SCmatrix[indexBarcode1][indexBarcode2][itrace],]
                            )
                        elif mode == "min":
                            SCmatrix[indexBarcode1][indexBarcode2][itrace] = np.nanmin(
                                [newdistance, SCmatrix[indexBarcode1][indexBarcode2][itrace],]
                            )

        self.SCmatrix = SCmatrix
        self.meanSCmatrix = np.nanmean(SCmatrix, axis=2)
        self.uniqueBarcodes = uniqueBarcodes

    def calculatesNmatrix(self):

        numberCells = self.SCmatrix.shape[2]

        if numberCells > 0:
            Nmatrix = np.sum(~np.isnan(self.SCmatrix), axis=2)
        else:
            numberBarcodes = self.SCmatrix.shape[0]
            Nmatrix = np.zeros((numberBarcodes, numberBarcodes))

        self.Nmatrix = Nmatrix

    def plotsAllmatrices(self, file):
        """
        Plots all matrices after analysis

        Parameters
        ----------
        file : str
            trace file name used for get output filenames.

        Returns
        -------
        None.

        """
        numberROIs = 1
        outputFileName = file.split('.')[0] + '_Matrix'
        clim = 2.2

        # plots PWD matrix
        # uses KDE
        plotMatrix(
            self.SCmatrix,
            self.uniqueBarcodes,
            self.pixelSize,
            numberROIs,
            outputFileName,
            self.logNameMD,
            figtitle="PWD matrix - KDE",
            mode="KDE",  # median or KDE
            clim=clim,
            cm="terrain",
            fileNameEnding="_PWDmatrixKDE.png",
        )  # need to validate use of KDE. For the moment it does not handle well null distributions

        # uses median
        plotMatrix(
            self.SCmatrix,
            self.uniqueBarcodes,
            self.pixelSize,
            numberROIs,
            outputFileName,
            self.logNameMD,
            figtitle="PWD matrix - median",
            mode="median",  # median or KDE
            clim=clim,
            cm="coolwarm",
            fileNameEnding="_PWDmatrixMedian.png",
        )  # need to validate use of KDE. For the moment it does not handle well null distributions

        # calculates and plots contact probability matrix from merged samples/datasets
        HiMmatrix, nCells = calculateContactProbabilityMatrix(
            self.SCmatrix, self.uniqueBarcodes, self.pixelSize, norm="nonNANs",
        )  # norm: nCells (default), nonNANs

        cScale = HiMmatrix.max()
        plotMatrix(
            HiMmatrix,
            self.uniqueBarcodes,
            self.pixelSize,
            numberROIs,
            outputFileName,
            self.logNameMD,
            figtitle="Hi-M matrix",
            mode="counts",
            clim=cScale,
            cm="coolwarm",
            fileNameEnding="_HiMmatrix.png",
        )

        # plots Nmatrix
        plotMatrix(
            self.Nmatrix,
            self.uniqueBarcodes,
            self.pixelSize,
            numberROIs,
            outputFileName,
            self.logNameMD,
            figtitle="N-matrix",
            mode="counts",
            clim=np.max(self.Nmatrix),
            cm="Blues",
            fileNameEnding="_Nmatrix.png",
        )

        plotDistanceHistograms(
            self.SCmatrix, self.pixelSize, outputFileName, self.logNameMD, mode="KDE", kernelWidth=0.25, optimizeKernelWidth=False
        )

    def save_matrices(self,file):

        outputFileName = file.split('.')[0] + '_Matrix'

        # saves output
        np.save(outputFileName + "_HiMscMatrix.npy", self.SCmatrix)
        np.savetxt(outputFileName + "_uniqueBarcodes.ecsv", self.uniqueBarcodes, delimiter=" ", fmt="%d")
        np.save(outputFileName + "_Nmatrix.npy", self.Nmatrix)

    def launch_analysis(self, file):
        """
        run analysis for a chromatin trace table.

        Returns
        -------
        None.

        """
        # outputFileName + "_mask:" + str(self.maskIdentifier) + "_ROI:" + str(self.nROI) + ".ecsv"

        # creates and loads trace table
        self.trace_table = chromatin_trace_table()
        self.trace_table.load(file)

        # decodes ROI

        # runs calculation of PWD matrix
        self.buildsdistanceMatrix("min")  # mean min last

        # calculates N-matrix: number of PWD distances for each barcode combination
        self.calculatesNmatrix()

        # runs plotting operations
        self.plotsAllmatrices(file)

        # saves matrix
        self.save_matrices(file)

    def run(self):

        # initializes sessionName, dataFolder, currentFolder
        label = "barcode"
        self.dataFolder, self.currentFolder  = initialize_module(self.param, module_name="build_matrix",label = label)

        # reads chromatin traces
        files = [x for x in glob.glob(self.dataFolder.outputFiles["buildsPWDmatrix"] + "_mask*" + label + ".ecsv")]

        if len(files) < 1:
            printLog("$ No chromatin trace table found to process!","WARN")
            return

        for file in files:
            self.launch_analysis(file)

        printLog(f"$ {len(files)} chromatin trace tables processed in {self.currentFolder}")
