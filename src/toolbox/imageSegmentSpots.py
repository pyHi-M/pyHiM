#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:54:49 2021

@author: marcnol


- loads a list of images from a folder
- segments image in 3D
- saves output in TIF

Steps:
    - defines run_parameters
    - loads iteratively images and segments volumes
    - saves output

"""


import argparse
import glob
import os
import sys

import numpy as np
from skimage import io
from tifffile import imsave
from tqdm import tqdm

from core.saving import save_image_as_blocks
from imageProcessing.segmentMasks import _segment_3d_volumes_by_thresholding


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images")
    parser.add_argument("-O", "--outputFile", help="Provide input file ")

    parser.add_argument("-M", "--mode", help="Mode: tif, fits, npy")
    parser.add_argument(
        "-W",
        "--wildcard",
        help="For instance '*tif'. If none is give, it will get all the tif files in the folder.",
    )
    parser.add_argument("-Z", "--zPlane", help="z-plane for 3D images")
    parser.add_argument(
        "--threshold_over_std",
        help="threshold_over_std for thresholding. Default = 2 (ie. 2% of max intensity range)",
    )
    parser.add_argument(
        "--area_min", help="area_min of each object, in pixels. Default = 3"
    )
    parser.add_argument(
        "--area_max", help="area_max of each object, in pixels. Default = 1000"
    )
    parser.add_argument("--nlevels", help="nlevels for deblending. Default = 32")
    parser.add_argument("--contrast", help="contrast for deblending. Default = 0.001")
    parser.add_argument(
        "--blockSizeXY",
        help="blockSizeXY for breaking image into blocks. Default = 256",
    )

    args = parser.parse_args()

    print("\n-------------------------------------------------------------------")
    run_parameters = {}

    if args.rootFolder:
        run_parameters["rootFolder"] = args.rootFolder
    else:
        print("\n> root_folder NOT FOUND, using PWD")
        run_parameters["rootFolder"] = os.getcwd()

    if args.outputFile:
        run_parameters["outputFile"] = (
            run_parameters["rootFolder"] + os.sep + args.outputFile
        )
    else:
        run_parameters["outputFile"] = None

    if args.mode:
        run_parameters["mode"] = args.mode
    else:
        run_parameters["mode"] = "tif"

    if args.wildcard:
        run_parameters["wildcard"] = args.wildcard
    else:
        run_parameters["wildcard"] = "None"

    if args.zPlane:
        run_parameters["zPlane"] = int(args.zPlane)
    else:
        run_parameters["zPlane"] = np.nan

    if args.threshold_over_std:
        run_parameters["threshold_over_std"] = int(args.threshold_over_std)
    else:
        run_parameters["threshold_over_std"] = 2

    if args.area_min:
        run_parameters["area_min"] = int(args.area_min)
    else:
        run_parameters["area_min"] = 3

    if args.area_max:
        run_parameters["area_max"] = int(args.area_max)
    else:
        run_parameters["area_max"] = 1000

    if args.nlevels:
        run_parameters["nlevels"] = int(args.nlevels)
    else:
        run_parameters["nlevels"] = 32

    if args.contrast:
        run_parameters["contrast"] = float(args.contrast)
    else:
        run_parameters["contrast"] = 0.001

    if args.blockSizeXY:
        run_parameters["blockSizeXY"] = float(args.blockSizeXY)
    else:
        run_parameters["blockSizeXY"] = 256

    print(
        "\n> Arguments parsed from command line list: {}\nNo problem found.".format(
            run_parameters
        )
    )

    return run_parameters


# =============================================================================
# MAIN
# =============================================================================


def main():

    # - defines run_parameters
    run_parameters = parse_arguments()

    # - gets list of images in folder using wildcard, *tif by default
    if "None" in run_parameters["wildcard"]:
        extension = run_parameters["mode"]
        search_string = run_parameters["rootFolder"] + os.sep + "*." + extension
    else:
        print("wildcard: {}".format(run_parameters["wildcard"]))
        search_string = (
            run_parameters["rootFolder"] + os.sep + run_parameters["wildcard"]
        )
        extension = run_parameters["wildcard"].split(".")[1]

    files_to_process = glob.glob(search_string)
    files_to_process.sort()

    if len(files_to_process) < 1:
        print("No images found!")
        sys.exit()
    else:
        print("Number of images to process: {}".format(len(files_to_process)))

    # sets expected image size
    img = io.imread(files_to_process[0]).squeeze()
    imgSize = img.shape
    print("Expected image sizes: {}".format(imgSize))

    # - loads iteratively images and sums
    i = 0
    threshold_over_std = run_parameters["threshold_over_std"]
    area_min = run_parameters["area_min"]
    area_max = run_parameters["area_max"]
    nlevels = run_parameters["nlevels"]
    contrast = run_parameters["contrast"]
    block_size_xy = run_parameters["blockSizeXY"]

    for file in tqdm(files_to_process):
        print("\nLoading and processing: {}".format(file))

        # loads file
        image_3d = io.imread(file).squeeze()

        print("Segmenting images...")
        binaryMask, newLabeledImage = _segment_3d_volumes_by_thresholding(
            image_3d,
            threshold_over_std=threshold_over_std,
            sigma=3,
            box_size=(32, 32),
            area_min=area_min,
            area_max=area_max,
            nlevels=nlevels,
            contrast=contrast,
            deblend_3d=True,
        )
        # breaks and saves image in 3D blocks
        save_image_as_blocks(newLabeledImage, file, block_size_xy=block_size_xy)

        # - save outputs
        file_name = file.split(".")[0]
        if run_parameters["outputFile"] is None:
            output_file = file_name + "_segmented"
        else:
            output_file = run_parameters["outputFile"] + "_" + str(i) + "_segmented"
            i += 1

        outfile = output_file + "." + extension
        print("\n> Saving image : {}".format(outfile))
        imsave(outfile, newLabeledImage)


if __name__ == "__main__":
    main()
