#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:21:02 2020

@author: marcnol

contains functions and classes needed for the analysis and plotting of HiM matrices

"""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import os
import glob
import json, csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.io import loadmat

from astropy.table import Table, vstack

from fileManagement import writeString2File
from alignBarcodesMasks import plotMatrix, plotDistanceHistograms, plotMatrix, calculateContactProbabilityMatrix
from alignBarcodesMasks import distributionMaximumKernelDensityEstimation

# =============================================================================
# CLASSES
# =============================================================================


class analysisHiMmatrix:
    """
    this class is used for loading data processed by processHiMmatrix.py 
    Main use is to produce paper quality figures of HiM matrices, 3-way interaction matrices and HiM matrix ratios
    """

    def __init__(self, runParameters, rootFolder="."):
        self.dataFolder = rootFolder + os.sep + "scHiMmatrices"
        self.runParameters = runParameters
        self.rootFolder = rootFolder
        self.data = []
        self.dataFiles = []
        self.folders2Load = []

    def loadData(self):
        """
        loads dataset

        Returns
        -------
        self.foldes2Load contains the parameters used for the processing of HiM matrices. 
        self.dataFiles dictionary containing the extensions needed to load data files
        self.data dictionary containing the datasets loaded
        """

        # loads datasets: parameter files
        fileNameListDataJSON = self.rootFolder + os.sep + self.runParameters["parametersFileName"]
        with open(fileNameListDataJSON) as json_file:
            ListData = json.load(json_file)

        datasetName = list(ListData.keys())[0]
        print("Dataset: {}".format(datasetName))

        outputFileName = (
            self.dataFolder
            + os.sep
            + datasetName
            + "_label:"
            + self.runParameters["label"]
            + "_action:"
            + self.runParameters["action"]
        )

        fileNameParametersJSON = outputFileName + "_parameters.json"
        with open(fileNameParametersJSON) as json_file:
            folders2Load = json.load(json_file)
        print("Loading parameter file:".format(fileNameParametersJSON))

        # Creates filenames to be loaded
        dataFiles = {}
        dataFiles["ensembleContactProbability"] = "_ensembleContactProbability.npy"
        dataFiles["SCmatrixCollated"] = "_SCmatrixCollated.npy"
        dataFiles["SClabeledCollated"] = "_SClabeledCollated.npy"

        if "3wayContacts_anchors" in ListData[datasetName]:
            for iAnchor in ListData[datasetName]["3wayContacts_anchors"]:
                newKey = "anchor:" + str(iAnchor - 1)
                dataFiles[newKey] = "_" + newKey + "_ensemble3wayContacts.npy"
        else:
            print("No anchors found")

        # loads datasets: numpy matrices
        data = {}
        for idataFile in dataFiles.keys():
            print("Loaded: {}: <{}>".format(idataFile,os.path.basename(outputFileName+dataFiles[idataFile])))
            data[idataFile] = np.load(outputFileName + dataFiles[idataFile]).squeeze()

        # loads datasets: lists
        runName = loadList(outputFileName + "_runName.csv")
        data["runName"] = runName
        print("Loaded runNames: {}".format(data["runName"]))

        data["uniqueBarcodes"] = loadList(outputFileName + "_uniqueBarcodes.csv")
        print("Loaded barcodes #: {}".format(data["uniqueBarcodes"]))

        print("Total number of cells loaded: {}".format(data["SCmatrixCollated"].shape[2]))
        print("Number Datasets loaded: {}".format(len(data["runName"])))

        # Exports data
        self.data = data
        self.dataFiles = dataFiles
        self.folders2Load = folders2Load
        self.ListData = ListData
        self.datasetName = datasetName

    # functions

    def plot2DMatrixSimple(
        self,
        ifigure,
        matrix,
        uniqueBarcodes,
        yticks,
        xticks,
        cmtitle="probability",
        cMin=0,
        cMax=1,
        cm="coolwarm",
        fontsize=12,
        colorbar=False,
        axisTicks=False,
        nCells=0,
        nDatasets=0,
        showTitle=False
    ):

        pos = ifigure.imshow(matrix, cmap=cm)  # colormaps RdBu seismic

        if showTitle:
            titleText="N = {} | n = {}".format(nCells,nDatasets)
            ifigure.title.set_text(titleText)

        # plots figure
        if xticks:
            ifigure.set_xlabel("barcode #", fontsize=fontsize)
            if not axisTicks:
                ifigure.set_xticklabels(())
            else:
                print("barcodes:{}".format(uniqueBarcodes))
                # ifigure.set_xticks(np.arange(matrix.shape[0]),uniqueBarcodes)
                ifigure.set_xticklabels(uniqueBarcodes)

        else:
            ifigure.set_xticklabels(())
        if yticks:
            ifigure.set_ylabel("barcode #", fontsize=fontsize)
            if not axisTicks:
                ifigure.set_yticklabels(())
            else:
                # ifigure.set_yticks(np.arange(matrix.shape[0]), uniqueBarcodes)
                ifigure.set_yticklabels(uniqueBarcodes)
        else:
            ifigure.set_yticklabels(())

        for xtick, ytick in zip(ifigure.xaxis.get_majorticklabels(), ifigure.yaxis.get_majorticklabels()):
            xtick.set_fontsize(fontsize)
            ytick.set_fontsize(fontsize)

        if colorbar:
            cbar = plt.colorbar(pos, ax=ifigure, fraction=0.046, pad=0.04)
            cbar.minorticks_on()
            cbar.set_label(cmtitle,fontsize=float(fontsize)*0.85)
            pos.set_clim(vmin=cMin, vmax=cMax)

        pos.set_clim(vmin=cMin, vmax=cMax)
        
        return pos

    def update_clims(self, cMin, cMax, axes):
        for ax in axes:
            ax.set_clim(vmin=cMin, vmax=cMax)



    def plot1Dprofile1Dataset(self,ifigure, anchor, iFigLabel, yticks, xticks):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        lwbase = plt.rcParams['lines.linewidth']
        thin, thick = lwbase / 2, lwbase * 3

        profile = self.data["ensembleContactProbability"][:,anchor-1] 
        x = np.linspace(0, profile.shape[0], num=profile.shape[0], endpoint=True)
        # f = interp1d(x, profile,kind = 'linear') # linear
        tck = interpolate.splrep(x, profile, s=0)
        xnew = np.linspace(0, profile.shape[0], num=100, endpoint=True)
        ynew = interpolate.splev(xnew, tck, der=0)
        if self.runParameters["splines"]:
            ifigure.plot(xnew, ynew, '-') # x, profile, 'o',
        else:
            ifigure.plot(x, profile, '-') # x, profile, 'o',

        ifigure.set_xlim([0,profile.shape[0]])
        ifigure.axvline(x=anchor-0.5, color=colors[4], lw=thick, alpha=0.5)
        ifigure.set_ylim([0,self.runParameters["cAxis"]])
        
        if xticks:
            ifigure.set_xlabel("barcode #", fontsize=self.runParameters['fontsize'])
            if not self.runParameters["axisTicks"]:
                ifigure.set_xticklabels(())
            else:
                ifigure.set_xticklabels(self.data['uniqueBarcodes'])
        else:
            ifigure.set_xticklabels(())

        if yticks:
            ifigure.set_ylabel("Probability", fontsize=self.runParameters['fontsize'])
            if not self.runParameters["axisTicks"]:
                ifigure.set_yticklabels(())
            else:
                ifigure.set_yticks([0, self.runParameters["cAxis"]/2, self.runParameters["cAxis"]])
        else:
            ifigure.set_yticklabels(())


    def nCellsLoaded(self):    
        if self.runParameters['action']=='labeled':
            cellswithLabel=[idx for idx, x in enumerate(self.data['SClabeledCollated']) if x>0]
            nCells=len(cellswithLabel)
        elif self.runParameters['action']=='unlabeled':         
            cellswithLabel=[idx for idx, x in enumerate(self.data['SClabeledCollated']) if x==0]
            nCells=len(cellswithLabel)
        else:
            nCells = self.data["SCmatrixCollated"].shape[2]
        print('nCells selected with label: {}'.format(nCells))
        return nCells

# =============================================================================
# FUNCTIONS
# =============================================================================

def plot1Dprofile2Datasets(ifigure, HiMdata1, HiMdata2,runParameters, anchor, iFigLabel, yticks, xticks,legend=False):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lwbase = plt.rcParams['lines.linewidth']
    thin, thick = lwbase / 2, lwbase * 3
    
    profile1 = HiMdata1.data["ensembleContactProbability"][:,anchor-1] 
    profile2 = HiMdata2.data["ensembleContactProbability"][:,anchor-1] 
    x = np.linspace(0, profile1.shape[0], num=profile1.shape[0], endpoint=True)
    tck1 = interpolate.splrep(x, profile1, s=0)
    tck2 = interpolate.splrep(x, profile2, s=0)
    xnew = np.linspace(0, profile1.shape[0], num=100, endpoint=True)
    ynew1 = interpolate.splev(xnew, tck1, der=0)
    ynew2 = interpolate.splev(xnew, tck2, der=0)
    if runParameters["splines"]:
        ifigure.plot(xnew, ynew1, '-',xnew, ynew2, '-') # x, profile, 'o',
    else:
        ifigure.plot(x, profile1, '-',x, profile2, '-') # x, profile, 'o',
    
    ifigure.set_xlim([0,profile1.shape[0]])
    ifigure.axvline(x=anchor-0.5, color=colors[4], lw=thick, alpha=0.5)
    ifigure.set_ylim([0,runParameters["cAxis"]])
    
    if xticks:
        ifigure.set_xlabel("barcode #", fontsize=runParameters['fontsize'])
        if not runParameters["axisTicks"]:
            ifigure.set_xticklabels(())
        else:
            ifigure.set_xticklabels(HiMdata1.data['uniqueBarcodes'])
    else:
        ifigure.set_xticklabels(())

    if yticks:
        ifigure.set_ylabel("Probability", fontsize=runParameters['fontsize'])
        if not runParameters["axisTicks"]:
            ifigure.set_yticklabels(())
        else:
            ifigure.set_yticks([0, runParameters["cAxis"]/2, runParameters["cAxis"]])
    else:
        ifigure.set_yticklabels(())
    
    if legend:
        ifigure.legend([HiMdata1.datasetName, HiMdata2.datasetName], loc='best')

def loadList(fileName):
    with open(fileName, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        runName = []
        for row in spamreader:
            # print(', '.join(row))
            if len(runName) > 0:
                runName.append(row)
            else:
                runName = row

    return runName


def attributesLabels2cells(SNDtable, ResultsTable, label="doc"):

    sortedSNDTable = SNDtable.group_by("MaskID #")
    listKeys = list(sortedSNDTable.groups.keys["MaskID #"].data)
    indexKey = [index for i, index in zip(listKeys, range(len(listKeys))) if i == label]
    
    # checks that there is at least one cell with the label
    if len(indexKey)>0:
    
        SNDTablewithLabel = sortedSNDTable.groups[indexKey[0]]
        print("\n>>> Matching labels")
        print("Found {} out of {} cells with {} in dataset".format(len(SNDTablewithLabel), len(sortedSNDTable), label))
    
        # sorts Results Table by ROI
        PWDTableSortedROI = ResultsTable.group_by("ROI #")
        CUIDsList = []
        CUIDs = Table()
        CUIDs["Cuid"]=[]
        
        print("ROIs to process: {}".format(PWDTableSortedROI.groups.keys))
        
        for ROI, group in zip(PWDTableSortedROI.groups.keys, PWDTableSortedROI.groups):
            # list of cellIDs in ROI
            cells2Process = group["CellID #"].data.compressed()
            cells2ProcessUID = group["Cuid"]
    
            # list of ROIs detected in SNDTablewithLabel
            ROIsinSNDTablewithLabel = list(SNDTablewithLabel.group_by("ROI #").groups.keys["ROI #"].data)
    
            # index of ROI within the keys of SNDTablewithLabel
            indexROIs = [index for i, index in zip(ROIsinSNDTablewithLabel, range(len(ROIsinSNDTablewithLabel))) if i == ROI["ROI #"]]
    
            # subtable of cells with label and ROI that we are looking for
            SNDTablewithLabelROI = SNDTablewithLabel.group_by("ROI #").groups[indexROIs[0]]
            cellswithLabel = list(SNDTablewithLabelROI["CellID #"].data)
    
            # finds which cell indeces in Table have label
            listofSelectedCells = [index for iCell, index in zip(cells2Process, range(len(cells2Process))) if iCell in cellswithLabel]
    
            if len(listofSelectedCells)>0:
                print('Detected {} cells in ROI {} with label'.format(len(listofSelectedCells),ROI["ROI #"]))
                if len(CUIDs) > 0:
                    # CUIDs = vstack([CUIDs, cells2ProcessUID[listofSelectedCells]])
                    # print('adding {} more cells'.format(len(cells2ProcessUID[listofSelectedCells])))
                    CUIDsList += list(cells2ProcessUID[listofSelectedCells].data.compressed())
                else:
                    CUIDsList = list(cells2ProcessUID[listofSelectedCells].data.compressed())
                    # CUIDs = cells2ProcessUID[listofSelectedCells]

            print(
                "Processed ROI # {}, found {} out of {} cells with {}".format(
                    ROI["ROI #"], len(listofSelectedCells), len(group), label
                )
            )
    
        # from list of CUIDs from cells that show label, I construct a binary vector of the same size as SCmatrix. Labeled cells have a 1.
        SClabeled = np.zeros(len(ResultsTable))
        # print('CUID list: {}'.format(CUIDsList2))
        # CUIDsList = CUIDs["Cuid"].data.compressed()
        # CUIDsList = CUIDsList2
        # checks that there are cells found with the label 
        if len(CUIDsList)>0:
            indexCellsWithLabel = [iRow for Row, iRow in zip(ResultsTable, range(len(ResultsTable))) if Row["Cuid"] in CUIDsList]
            SClabeled[indexCellsWithLabel] = 1
        else:
            SClabeled, CUIDsList = [], []
            
        return SClabeled, CUIDsList
    else:
        # otherwise returns an empty list
        print("Warning: No cell with a mask labeled <{}> was found".format(label))
        return [], []
        

def loadsSCdata(ListData, datasetName, p):
    """
    loads SC datasets from a dict of folders (ListData)

    Parameters
    ----------
    ListData : dict
        dict of folders with data that can be loaded.
    dataset2Load : int, optional
        The item in the dictionary that will be loaded. The default is 3.

    Returns
    -------
    SCmatrixCollated : list of np arrays nBarcodes x nBarcodes x nCells
        Cummulative SC PWD matrix.
    uniqueBarcodes : list of np arrays 
        containing the barcode identities for each matrix.
    buildsPWDmatrixCollated : list of Tables
        Tables with all the data for cells and barcodes used to produce SCmatrixCollated.

    """
    # tags2process = list(ListData.keys())
    print("Dataset to load: {}\n\n".format(list(ListData.keys())[0]))

    SCmatrixCollated, uniqueBarcodes = [], []
    buildsPWDmatrixCollated, runName, SClabeledCollated = [], [], []

    for rootFolder in ListData[datasetName]["Folders"]:
        # [finds sample name]
        runName.append(os.path.basename(os.path.dirname(rootFolder)))

        # [makes list of files with Tables to load]
        # tries to load files from newer version of proceesingPipeline.py
        files2Process = glob.glob(rootFolder + "/buildsPWDmatrix_order*ROI*.ecsv")
        if len(files2Process) == 0:
            # it resorts to old format
            files2Process = glob.glob(rootFolder + "/buildsPWDmatrix_*ROI*.ecsv")
        else:
            print("Found {} ECSV files in {}".format(len(files2Process), rootFolder))

        # checks that something was found
        if len(files2Process) > 0:
            print(">>> Loading {} results tables".format(len(files2Process)))

            # [initializes variables]
            buildsPWDmatrix = Table()
            fileOrder, fileOrderStamp, fileTimeStamp = (
                np.zeros(len(files2Process), dtype=int),
                np.zeros(len(files2Process), dtype=int),
                np.zeros(len(files2Process)),
            )

            # [finds what order Table files should be loaded to agree with order in buildsPWDmatrix_HiMscMatrix.npy]
            for fileName, ifileName in zip(files2Process, range(len(files2Process))):
                if "_order" in fileName:
                    for isplit in fileName.split("_"):
                        if "order" in isplit:
                            fileOrderStamp[ifileName] = int(isplit.split(":")[1])
                            print("order {}= {}--> {}".format(ifileName, os.path.basename(fileName), fileOrderStamp[ifileName]))
                    fileTimeStamp[ifileName] = os.path.getmtime(fileName)
                    choosingTimeStamp = False
                    
                else:
                    fileTimeStamp[ifileName] = os.path.getmtime(fileName)
                    choosingTimeStamp = True

            if choosingTimeStamp:
                fileOrder = np.argsort(fileTimeStamp).astype(int)
            else:
                fileOrder = np.argsort(fileOrderStamp).astype(int)
                
            # print('FileOrder: {}'.format(fileOrder))
            # [loads buildsPWDmatrix Tables]
            for ifileName in range(len(files2Process)):
                fileName = files2Process[fileOrder[ifileName]]
                newbuildsPWDmatrix = Table.read(fileName, format="ascii.ecsv")  # ascii.ecsv
                buildsPWDmatrix = vstack([buildsPWDmatrix, newbuildsPWDmatrix])
                print(
                    "[{}:{}:{}] From {}, Read: {} cells, Cummulative: {} cells".format(
                        ifileName,
                        fileOrder[ifileName],
                        fileTimeStamp[fileOrder[ifileName]],
                        os.path.basename(fileName),
                        len(newbuildsPWDmatrix),
                        len(buildsPWDmatrix),
                    )
                )

            # [loads SNDassignedCells.ecsv files if available]
            fileNameSNDassignedCells = os.path.dirname(rootFolder) + os.sep + "segmentedObjects/SNDassignedCells.ecsv"
            if os.path.exists(fileNameSNDassignedCells):
                print('Reading and processing: {}'.format(fileNameSNDassignedCells))
                SNDassignedCells = Table.read(fileNameSNDassignedCells, format="ascii.ecsv")

                # checks that table is not empty
                if len(SNDassignedCells)>0: 
                    # attributes masks to single cells
                    SClabeled, CUIDsList = attributesLabels2cells(SNDassignedCells, buildsPWDmatrix, label=p["label"])
                    
                    # checks that at least one cell was found to have the label
                    if len(SClabeled)==0:
                        # if not available it makes a mock SClabeled matrix so that pipeline always works
                        # SClabeled = np.ones(len(buildsPWDmatrix)).astype(int)
                        SClabeled = np.zeros(len(buildsPWDmatrix)).astype(int)
            
            else:
                # if not available it makes a mock SClabeled matrix so that pipeline always works
                # SClabeled = np.ones(len(buildsPWDmatrix)).astype(int)
                SClabeled = np.zeros(len(buildsPWDmatrix)).astype(int)
 
            SClabeledCollated.append(SClabeled)

            # [loads and accumulates barcodes and scHiM matrix]
            fileNamMatrix = rootFolder + os.sep + "buildsPWDmatrix_HiMscMatrix.npy"
            fileNameBarcodes = rootFolder + os.sep + "buildsPWDmatrix_uniqueBarcodes.ecsv"

            if os.path.exists(fileNamMatrix):
                SCmatrix1 = np.load(fileNamMatrix)
                SCmatrixCollated.append(SCmatrix1)
            else:
                print("*** Error: could not find {}".format(fileNamMatrix))

            if os.path.exists(fileNameBarcodes):
                uniqueBarcodes.append(np.loadtxt(fileNameBarcodes).astype(int))
            else:
                print("*** Error: could not find {}".format(fileNameBarcodes))

            buildsPWDmatrixCollated.append(buildsPWDmatrix)

            print("\n>>>Merging rootFolder: {}".format(rootFolder))
            print("Cells added after merge: {}\n".format(SCmatrix1.shape[2]))
        else:
            print("No file detected in the folder you provide: {}".format(rootFolder + "/buildsPWDmatrix_*ROI*.ecsv"))
    print("{} datasets loaded\n".format(len(SCmatrixCollated)))

    return (
        SCmatrixCollated,
        uniqueBarcodes,
        buildsPWDmatrixCollated,
        runName,
        SClabeledCollated,
    )

def loadsSCdataMATLAB(ListData, datasetName, p):


    print("Dataset to load: {}\n\n".format(list(ListData.keys())[0]))

    SCmatrixCollated, uniqueBarcodes = [], []
    runName, SClabeledCollated = [], []

    for rootFolder in ListData[datasetName]["Folders"]:
        # [finds sample name]
        runName.append(os.path.basename(os.path.dirname(rootFolder)))
        
        # [loads and accumulates barcodes and scHiM matrix]
        fileNameMatrix = rootFolder + os.sep + "HiMscMatrix.mat"
        fileNameBarcodes = rootFolder + os.sep + "buildsPWDmatrix_uniqueBarcodes.ecsv"

        # loads barcodes
        if os.path.exists(fileNameBarcodes):
            uniqueBarcodes.append(np.loadtxt(fileNameBarcodes).astype(int))
            print(">>> Loaded {}".format(fileNameMatrix))
        else:
            print("*** Error: could not find {}".format(fileNameBarcodes))

        # loads SC matrix
        if os.path.exists(fileNameMatrix):
            data=loadmat(fileNameMatrix)
            SCmatrix1 =data['distanceMatrixCumulative']
            # print(">>> SC matrix 1 shape: {}".format(SCmatrix1.shape))
            # SCmatrix2=SCmatrix1[uniqueBarcodes[0]-1,uniqueBarcodes[0]-1,:]
            SCmatrixCollated.append(SCmatrix1)
            print(">>> Loaded: {}\n SC matrix shape: {}".format(fileNameMatrix,SCmatrix1.shape))
        else:
            print("*** Error: could not find {}".format(fileNameMatrix))

       # loads cell attributes
        cellAttributesMatrix = data['cellAttributesMatrix']
        ResultsTable = cellAttributesMatrix[0,:]

        SClabeled = np.zeros(len(ResultsTable))
        indexCellsWithLabel = [iRow for iRow, Row in enumerate(ResultsTable) if Row > 0 ]
        SClabeled[indexCellsWithLabel] = 1
        SClabeledCollated.append(SClabeled)

        print("\n>>>Merging rootFolder: {}".format(rootFolder))
        print("Cells added after merge: {}\n".format(SCmatrix1.shape[2]))

    print("{} datasets loaded\n".format(len(SCmatrixCollated)))

    return (
        SCmatrixCollated,
        uniqueBarcodes,
        runName,
        SClabeledCollated,
    )


def listsSCtoKeep(p, mask):
    print('{}:{}'.format(p['label'],p['action']))
    if p["action"] == "all":
        try:
            cells2Plot = range(len(mask))
        except TypeError:
            print(mask)
            cells2Plot = range(mask.shape[0])
    elif p["action"] == "labeled":
        a = [i for i in range(len(mask)) if mask[i] == 1]
        cells2Plot = a
    elif p["action"] == "unlabeled":
        a = [i for i in range(len(mask)) if mask[i] == 0]
        cells2Plot = a

    return cells2Plot


def normalizeMatrix(SCmatrix_wt):
    SCmatrix_wt_normalized = SCmatrix_wt
    nBins = SCmatrix_wt.shape[0]

    for iRow in range(nBins):
        rowSum = np.sum(SCmatrix_wt[iRow, :])
        for iCol in range(nBins):
            SCmatrix_wt_normalized[iRow, iCol] = SCmatrix_wt_normalized[iRow, iCol] / rowSum
            SCmatrix_wt_normalized[iCol, iRow] = SCmatrix_wt_normalized[iCol, iRow] / rowSum
    return SCmatrix_wt_normalized


def plotsEnsemble3wayContactMatrix(
    SCmatrixCollated, uniqueBarcodes, anchors, sOut, runName, iListData, p, fileNameMD="tmp.md", datasetName=""
):

    # combines matrices from different embryos and calculates integrated contact probability matrix

    SCmatrixAllDatasets = []  # np.zeros((nBarcodes,nBarcodes))
    for iSCmatrixCollated, iuniqueBarcodes, mask, iTag in zip(SCmatrixCollated, uniqueBarcodes, p["SClabeledCollated"], runName):
        cells2Plot = listsSCtoKeep(p, mask)
        if len(cells2Plot)>0:
            if max(cells2Plot) > iSCmatrixCollated.shape[2]:
                print(
                    "Error: max in cells2plot {} in dataset {} is larger than the number of available cells {}".format(
                        max(cells2Plot), iTag, iSCmatrixCollated.shape[2]
                    )
                )
            else:
                if len(SCmatrixAllDatasets) > 0:
                    SCmatrixAllDatasets = np.concatenate((SCmatrixAllDatasets, iSCmatrixCollated[:, :, cells2Plot]), axis=2)
                else:
                    SCmatrixAllDatasets = iSCmatrixCollated[:, :, cells2Plot]
    
                commonSetUniqueBarcodes = iuniqueBarcodes
        else:
            print('Dataset: {} - {}  did not have any cell to plot'.format(datasetName,iTag))
            
    # print(commonSetUniqueBarcodes)
    for anchor in anchors:
        print("nCells processed: {}".format(SCmatrixAllDatasets.shape[2]))
        SCmatrix = calculate3wayContactMatrix(
            SCmatrixAllDatasets,
            uniqueBarcodes,
            p["pixelSize"],
            anchor,
            sOut,
            threshold=iListData["ContactProbability_distanceThreshold"],
            norm="nonNANs",
        )  # norm: nonNANs (default)

        # outputFileName = p['outputFolder'] + os.sep + datasetName + "_Cells:" + p['action'] + "_ensemble3wayContacts"
        outputFileName = (
            p["outputFolder"] + os.sep + datasetName + "_label:" + p["label"] + "_action:" + p["action"] + "_ensemble3wayContacts"
        )

        outputFileName += "_anchor_" + str(anchor)
        writeString2File(fileNameMD, "![]({})\n".format(outputFileName + "_HiMmatrix.png"), "a")

        cScale = np.max(SCmatrix)
        # print(cScale)
        # print(SCmatrix.shape)

        plotMatrix(
            SCmatrix,
            uniqueBarcodes,
            p["pixelSize"],
            cm=iListData["ContactProbability_cm"],
            outputFileName=outputFileName,
            clim=cScale,
            cMin=iListData["ContactProbability_cmin"],
            figtitle="3way contacts",
            cmtitle=sOut,
            nCells=0,
            mode="counts",
        )  # twilight_shifted_r

        # saves matrices as individual files for further plotting
        rootOutputFileName = (
            p["outputFolder"]
            + os.sep
            + datasetName
            + "_label:"
            + p["label"]
            + "_action:"
            + p["action"]
            + "_anchor:"
            + str(anchor)
        )
        np.save(rootOutputFileName + "_ensemble3wayContacts.npy", SCmatrix)


def calculate3wayContactMatrix(iSCmatrixCollated, iuniqueBarcodes, pixelSize, anchor, sOut, threshold=0.25, norm="nonNANs"):

    nX = nY = iSCmatrixCollated.shape[0]
    SCmatrix = np.zeros((nX, nY))

    # transform distance matrix from pixel to µm
    mat = pixelSize * iSCmatrixCollated

    # print(nX, nY)
    for bait1 in range(nX):
        for bait2 in range(nY):
            if bait1 == bait2:
                continue

            # print("current bait1", bait1, "bait2", bait2)
            n_contacts, n_nonNaN = getMultiContact(mat, anchor, bait1, bait2, threshold)
            if sOut == "Counts":
                SCmatrix[bait1, bait2] = n_contacts
            elif sOut == "Probability":
                SCmatrix[bait1, bait2] = n_contacts / n_nonNaN
            else:
                print("Unexpected sOut.")
                return -1

            # print(n_contacts / n_nonNaN)
            # print(type(n_contacts), type(n_nonNaN))

    SCmatrix[np.isnan(SCmatrix)] = 0  # set NaN to zero
    return SCmatrix


def getMultiContact(mat, anchor, bait1, bait2, threshold):
    """
    Input:
    mat        : pwd matrix, including only the bins of used RTs
    anchor     : anchor bin
    bait1      : bait bin #1
    bait2      : bait bin #2
    threshold  : contact threshold
    Output:
    n_contacts : number of contacts between bins anchor, bait1, and bait2
    n_nonNaN   : number of cells where the distances anchor-bait1 and anchor-bait2 are present
    """

    A = mat[anchor, bait1, :]
    B = mat[anchor, bait2, :]

    # get fraction of points in quadrant
    n1 = np.sum((A < threshold) & (B < threshold))
    totN = np.sum((~np.isnan(A)) & (~np.isnan(B)))

    return n1, totN


def plotsSinglePWDmatrices(
    SCmatrixCollated, uniqueBarcodes, runName, iListData, p, fileNameMD="tmp.md", datasetName="",
):
    # plots distance matrix for each dataset
    for iSCmatrixCollated, iuniqueBarcodes, iTag, mask in zip(SCmatrixCollated, uniqueBarcodes, runName, p["SClabeledCollated"]):
        outputFileName = p["outputFolder"] + os.sep + iTag + "_Cells:" + p["action"] + "_PWDmatrix"

        # selects cels according to label
        cells2Plot = listsSCtoKeep(p, mask)

        plotMatrix(
            iSCmatrixCollated,
            iuniqueBarcodes,
            p["pixelSize"],
            outputFileName=outputFileName,
            figtitle="PWD:" + datasetName+ iTag,
            cm=iListData["PWD_cm"],
            clim=iListData["PWD_clim"],
            mode=iListData["PWD_mode"],
            nCells=iSCmatrixCollated.shape[2],
            cells2Plot=cells2Plot,
        )  # twilight_shifted_r 1.4, mode: median KDE coolwarm terrain
        writeString2File(fileNameMD, "![]({})\n".format(outputFileName + "_HiMmatrix.png"), "a")


def plotsInversePWDmatrix(SCmatrixCollated, uniqueBarcodes, runName, iListData, p, fileNameMD, datasetName=""):
    # plots inverse distance matrix for each dataset
    for iSCmatrixCollated, iuniqueBarcodes, iTag, mask in zip(SCmatrixCollated, uniqueBarcodes, runName, p["SClabeledCollated"]):
        outputFileName = p["outputFolder"] + os.sep + iTag + "_Cells:" + p["action"] + "_invPWDmatrix"

        # selects cels according to label
        cells2Plot = listsSCtoKeep(p, mask)
        # print('Dataset {} cells2plot: {}'.format(iTag,cells2Plot))

        plotMatrix(
            iSCmatrixCollated,
            iuniqueBarcodes,
            p["pixelSize"],
            cm=iListData["iPWD_cm"],
            outputFileName=outputFileName,
            clim=iListData["iPWD_clim"],
            mode=iListData["iPWD_mode"],
            figtitle="inverse PWD:" + datasetName+ iTag,
            cmtitle="inverse distance, 1/nm",
            inverseMatrix=True,
            nCells=iSCmatrixCollated.shape[2],
            cells2Plot=cells2Plot,
        )  # twilight_shifted_r, mode: median KDE
        writeString2File(fileNameMD, "![]({})\n".format(outputFileName + "_HiMmatrix.png"), "a")


def plotsSingleContactProbabilityMatrix(
    SCmatrixCollated, uniqueBarcodes, runName, iListData, p, fileNameMD="tmp.md", datasetName="",
):
    # Plots contact probability matrices for each dataset

    for iSCmatrixCollated, iuniqueBarcodes, iTag, mask in zip(SCmatrixCollated, uniqueBarcodes, runName, p["SClabeledCollated"]):
        # selects cels according to label
        cells2Plot = listsSCtoKeep(p, mask)

        if not cells2Plot:
            break
        
        if max(cells2Plot) > iSCmatrixCollated.shape[2]:
            print(
                "Error with range in cells2plot {} as it is larger than the number of available cells {}".format(
                    max(cells2Plot), iSCmatrixCollated.shape[2]
                )
            )
        else:
            SCmatrix, nCells = calculateContactProbabilityMatrix(
                iSCmatrixCollated[:, :, cells2Plot],
                iuniqueBarcodes,
                p["pixelSize"],
                threshold=iListData["ContactProbability_distanceThreshold"],
                norm="nonNANs",
            )  # norm: nCells (default), nonNANs
            outputFileName = p["outputFolder"] + os.sep + datasetName+ iTag + "_Cells:" + p["action"] + "_contactProbability"

            print("Dataset {} cells2plot: {}".format(iTag, cells2Plot))
            cScale = SCmatrix.max() / iListData["ContactProbability_scale"]

            plotMatrix(
                SCmatrix,
                iuniqueBarcodes,
                p["pixelSize"],
                cm=iListData["ContactProbability_cm"],
                outputFileName=outputFileName,
                cMin=iListData["ContactProbability_cmin"],
                clim=cScale,
                figtitle="HiM:" + datasetName+ iTag,
                cmtitle="probability",
                nCells=nCells,
                cells2Plot=cells2Plot,
            )  # twilight_shifted_r terrain coolwarm
            writeString2File(fileNameMD, "![]({})\n".format(outputFileName + "_HiMmatrix.png"), "a")


def fusesSCmatrixCollatedFromDatasets(SCmatrixCollated, uniqueBarcodes, p, runName, iListData):
    # combines matrices from different embryos and calculates integrated contact probability matrix
    
    SCmatrixAllDatasets = []
    cells2Plot = []
    NcellsTotal=0

    for iSCmatrixCollated, iuniqueBarcodes, mask, iTag in zip(SCmatrixCollated, uniqueBarcodes, p["SClabeledCollated"], runName):
        NcellsTotal+=mask.shape[0]
        # selects cels according to label
        cells2Plot = listsSCtoKeep(p, mask)

        if len(cells2Plot)>0:
            if max(cells2Plot) > iSCmatrixCollated.shape[2]:
                print(
                    "Error: max in cells2plot {} in dataset {} is larger than the number of available cells {}".format(
                        max(cells2Plot), iTag, iSCmatrixCollated.shape[2]
                    )
                )
            else:
                if len(SCmatrixAllDatasets) > 0:
                    SCmatrixAllDatasets = np.concatenate((SCmatrixAllDatasets, iSCmatrixCollated[:, :, cells2Plot]), axis=2)
                else:
                    SCmatrixAllDatasets = iSCmatrixCollated[:, :, cells2Plot]
    
                commonSetUniqueBarcodes = iuniqueBarcodes
                
                
    if p["saveMatrix"]:
        # write out the ensemble PWD map
        Nbarcodes = SCmatrixAllDatasets.shape[0]
        pixelSize = p["pixelSize"]
        meanSCmatrix = np.zeros((Nbarcodes, Nbarcodes))
        for bin1 in range(Nbarcodes):
            for bin2 in range(Nbarcodes):
                if bin1 != bin2:
                    (maximumKernelDistribution, _, _, _,) = distributionMaximumKernelDensityEstimation(
                        SCmatrixAllDatasets, bin1, bin2, pixelSize, optimizeKernelWidth=False)
                    meanSCmatrix[bin1, bin2] = maximumKernelDistribution
        
        outputFileName=p["outputFolder"] + os.sep + "CombinedMatrix_PWD_KDE" + ":" + list(iListData.keys())[0] + "_Cells:" + p["action"] + ".dat"
        
        print(">>> Saving fused SCmatrix to {}".format(outputFileName))
    
        np.savetxt(
            outputFileName,
            meanSCmatrix,
            fmt="%.4f",
            delimiter=" ",
            newline="\n",
            header="Combined pairwise distance map (kernel density estimator)",
            footer="",
            comments="# ",
            encoding=None,
        )
                
    return SCmatrixAllDatasets, commonSetUniqueBarcodes, cells2Plot, NcellsTotal
            
def plotsEnsembleContactProbabilityMatrix(
    SCmatrixCollated, uniqueBarcodes, runName, iListData, p, fileNameMD="tmp.md", datasetName="",
):

    # combines matrices from different embryos and calculates integrated contact probability matrix    
    SCmatrixAllDatasets, commonSetUniqueBarcodes, cells2Plot, NcellsTotal = fusesSCmatrixCollatedFromDatasets(SCmatrixCollated, uniqueBarcodes, p, runName, iListData)

    print("nCells selected / processed: {}/{}".format(SCmatrixAllDatasets.shape[2],NcellsTotal))

    SCmatrix, nCells = calculateContactProbabilityMatrix(
        SCmatrixAllDatasets,
        commonSetUniqueBarcodes,
        p["pixelSize"],
        threshold=iListData["ContactProbability_distanceThreshold"],
        norm="nonNANs",
    )  # norm: nCells (default), nonNANs
    cScale = SCmatrix.max() / iListData["ContactProbability_scale"]
    outputFileName = p["outputFolder"] + os.sep + datasetName + "_Cells:" + p["action"] + "_ensembleContactProbability"
    writeString2File(fileNameMD, "![]({})\n".format(outputFileName + "_HiMmatrix.png"), "a")

    plotMatrix(
        SCmatrix,
        commonSetUniqueBarcodes,
        p["pixelSize"],
        cm=iListData["ContactProbability_cm"],
        outputFileName=outputFileName,
        clim=cScale,
        cMin=iListData["ContactProbability_cmin"],
        figtitle="HiM counts",
        cmtitle="probability",
        nCells=nCells,
    )  # twilight_shifted_r

    np.savetxt(
        p["outputFolder"] + os.sep + "CombinedMatrix" + ":" + list(iListData.keys())[0] + "_Cells:" + p["action"] + ".dat",
        SCmatrix,
        fmt="%.4f",
        delimiter=" ",
        newline="\n",
        header="Combined contact probability matrix",
        footer="",
        comments="# ",
        encoding=None,
    )

    np.savetxt(
        p["outputFolder"] + os.sep + "UniqueBarcodes" + ":" + list(iListData.keys())[0] + "_Cells:" + p["action"] + ".dat",
        commonSetUniqueBarcodes,
        fmt="%.4f",
        delimiter=" ",
        newline="\n",
        header="unique barcodes",
        footer="",
        comments="# ",
        encoding=None,
    )

    return SCmatrix, commonSetUniqueBarcodes

