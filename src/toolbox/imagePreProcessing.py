#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:44:47 2021

@author: marcnol

- loads a list of images from a folder
- re-scales exposures
- removes inhomogeneous background
- adjusts image levels
- saves output in TIF

Steps:
    - defines run_parameters
    - gets list of images in folder using wildcard, *tif by default
    - loads iteratively images and applies operations described above
    - saves output

"""


import argparse
import glob
import multiprocessing
import os
import sys
from datetime import datetime

import numpy as np
from dask.distributed import Client, LocalCluster
from skimage import exposure, io
from tifffile import imsave
from tqdm import tqdm

from core.saving import save_image_as_blocks
from imageProcessing.imageProcessing import (
    _remove_inhomogeneous_background,
    image_adjust,
)


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
        "--lower_threshold", help="lower_threshold for adjusting levels"
    )
    parser.add_argument(
        "--higher_threshold", help="higher_threshold for adjusting levels"
    )
    parser.add_argument(
        "--blockSizeXY",
        help="blockSizeXY for breaking image into blocks. Default = 256",
    )
    parser.add_argument("--parallel", help="Runs in parallel mode", action="store_true")

    args = parser.parse_args()

    print("\n--------------------------------------------------------------------")
    run_parameters = {}

    if args.rootFolder:
        run_parameters["rootFolder"] = args.rootFolder
    else:
        print("\n> rootFolder NOT FOUND, using PWD")
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
        run_parameters["zPlane"] = 0

    if args.lower_threshold:
        run_parameters["lower_threshold"] = float(args.lower_threshold)
    else:
        run_parameters["lower_threshold"] = 0.9

    if args.higher_threshold:
        run_parameters["higher_threshold"] = float(args.higher_threshold)
    else:
        run_parameters["higher_threshold"] = 0.9999

    if args.blockSizeXY:
        run_parameters["blockSizeXY"] = float(args.blockSizeXY)
    else:
        run_parameters["blockSizeXY"] = 256

    if args.parallel:
        run_parameters["parallel"] = args.parallel
    else:
        run_parameters["parallel"] = False

    print(
        "\n> Arguments parsed from command line list: {}\nNo problem found.".format(
            run_parameters
        )
    )

    return run_parameters


def lauch_dask_scheduler(requested_nb_nodes, maximum_load=0.6, memory_per_worker=2000):
    number_cores_available = multiprocessing.cpu_count()

    # we want at least 2 GB per worker
    _, _, free_m = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])

    max_number_threads = int(
        np.min([number_cores_available * maximum_load, free_m / memory_per_worker])
    )

    n_threads = int(np.min([max_number_threads, requested_nb_nodes]))

    print("Go to http://localhost:8787/status for information on progress...")

    cluster = LocalCluster(
        n_workers=n_threads,
        # processes=True,
        # threads_per_worker=1,
        # memory_limit='2GB',
        # ip='tcp://localhost:8787',
    )
    client = Client(cluster)

    return client, cluster


# =============================================================================
# MAIN
# =============================================================================


def main():
    begin_time = datetime.now()

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
    block_size_xy = run_parameters["blockSizeXY"]

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

    if run_parameters["parallel"]:
        client, cluster = lauch_dask_scheduler(6)

    # - loads iteratively images and sums
    i = 0
    for file in tqdm(files_to_process):
        # print("Loading and processing: {}".format(file))

        # loads file
        images = []
        raw_image = io.imread(file).squeeze()
        images.append(raw_image)

        # breaks and saves original image as 3D blocks
        save_image_as_blocks(raw_image, file, block_size_xy=block_size_xy)

        # autoscales exposures
        images = [exposure.rescale_intensity(x, out_range=(0, 1)) for x in images]

        # removes inhomogeneous background
        print("\nRemoving inhomogeneous background...")
        images = [_remove_inhomogeneous_background(x) for x in images]

        # rescales grey levels
        print("\nRescaling grey levels...")
        images = [
            image_adjust(
                x,
                lower_threshold=run_parameters["lower_threshold"],
                higher_threshold=run_parameters["higher_threshold"],
            )[0]
            for x in images
        ]

        # saves output
        file_name = file.split(".")[0]
        if run_parameters["outputFile"] is None:
            output_file = file_name + "_preProcessed"
        else:
            output_file = run_parameters["outputFile"] + "_" + str(i) + "_preProcessed"
            i += 1

        for index, image in enumerate(images):
            outfile = output_file + "_index:" + str(index) + "." + extension
            imsave(outfile, image)
            print("\n> Saving image : {}".format(outfile))

    if run_parameters["parallel"]:
        cluster.close()
        client.close()

    print("Elapsed time: {}".format(datetime.now() - begin_time))


if __name__ == "__main__":
    main()
