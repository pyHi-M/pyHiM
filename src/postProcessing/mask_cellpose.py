#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:05:08 2023

@author: marcnol

Usage:

    $ ls scan_*ROI.tif | mask_cellpose.py --gpu

or just for a single file

    mask_cellpose.py --input scan_001_DAPI_001_ROI.tif --gpu


"""
import argparse
import json
import os
import select
import subprocess
import sys

import numpy as np
from cellpose import core, models
from cellpose.io import imread
from scipy.ndimage import shift as shift_image
from tifffile import imwrite


def parseArguments():
    parser = argparse.ArgumentParser(
        description="""mask_cellpose.py will segment a TIF file using predermined parameters using cellpose. \n These parameters can be
        changed using arguments. See below """,
        epilog="""All is well that ends well.""",
    )

    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument(
        "--gpu", help="If used it will use gpu mode", action="store_true"
    )
    parser.add_argument(
        "--api",
        help="It will call the CLI command instead of the API (which sometimes crashes). Default = False",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cellprob", type=float, default=-8, help="cellprob threshold. Default = -8."
    )
    parser.add_argument("--flow", help="flow threshold. Default = 10.")
    parser.add_argument("--stitch", help="stitch threshold. Default = 0.1.")
    parser.add_argument("--diam", help="diameter. Default = 50.")
    parser.add_argument(
        "--model", help="pretrained_model to use for running. Default: “cyto”."
    )
    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    p = {}

    args = parser.parse_args()

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.gpu:
        p["gpu"] = [True, None]
    else:
        p["gpu"] = [False, None]

    if args.api:
        p["cli"] = False
    else:
        p["cli"] = True

    p["cellprob"] = float(args.cellprob)

    if args.flow:
        p["flow"] = float(args.flow)
    else:
        p["flow"] = 10

    if args.stitch:
        p["stitch"] = float(args.stitch)
    else:
        p["stitch"] = 0.1

    if args.diam:
        p["diam"] = float(args.diam)
    else:
        p["diam"] = 50

    if args.diam:
        p["model"] = str(args.model)
    else:
        p["model"] = "cyto"

    p["files"] = []
    if args.pipe:
        p["pipe"] = True
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.0,
        )[0]:
            p["files"] = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        p["pipe"] = False
        p["files"] = [p["input"]]

    return p


def run_cellpose_api(
    image_path, diam, cellprob, flow, stitch, gpu=[False, None], pretrained_model="cyto"
):
    # model_type='cyto' or 'nuclei' or 'cyto2'
    if gpu[0]:
        model = models.Cellpose(gpu=gpu[1], model_type=pretrained_model)
    else:
        model = models.Cellpose(model_type=pretrained_model)

    # list of files
    files = [image_path]

    imgs = [imread(f) for f in files]

    # define CHANNELS to run segmentation on
    channels = [[0, 0]]

    # runs model
    masks, flows, styles, diams = model.eval(
        imgs,
        channels=channels,
        diameter=diam,
        cellprob_threshold=cellprob,
        flow_threshold=flow,
        stitch_threshold=stitch,
        batch_size=8,
    )

    return masks[0]


def run_cellpose(
    image_path,
    diam,
    cellprob,
    flow,
    stitch,
    folder_destination="segmentedObjects",
    gpu=[False, None],
    pretrained_model="cyto",
):
    command = (
        "cellpose --verbose "
        + f"--image_path {image_path} --no_npy --save_tif "
        + f"--chan 0 --diameter {diam} "
        + f"--stitch_threshold {stitch} "
        + f"--flow_threshold {flow} "
        + f"--cellprob_threshold {cellprob} "
        + f"--pretrained_model {pretrained_model} "
    )

    if gpu[0]:
        command = command + " --use_gpu"

    print(f"> will run: {command}")
    subprocess.run(command, shell=True)

    mask_name = image_path.split(".")[0] + "_cp_masks.tif"  # "_seg.npy"
    print(f"> Reads TIF mask image: {mask_name}")

    if os.path.exists(mask_name):
        img = imread(mask_name)
        return img


def process_images(
    cellprob=-8,
    flow=10,
    stitch=0.1,
    diam=50,
    files=list(),
    gpu=[False, None],
    cli=False,
    pretrained_model="cyto",
):
    print(
        f"Parameters: diam={diam} | cellprob={cellprob} | flow={flow} | stitch={stitch}\n"
    )
    params = load_params()
    folder_mask_2d = (
        params["common"]["segmentedObjects"].get("mask_2d_folder", "mask_2d")
        + os.sep
        + "data"
    )
    folder_mask_3d = (
        params["common"]["segmentedObjects"].get("mask_3d_folder", "mask_3d")
        + os.sep
        + "data"
    )
    try:
        os.makedirs(folder_mask_2d)
        os.makedirs(folder_mask_3d)
    except FileExistsError:
        print(">>> Output folder exists")

    if len(files) > 0:
        print(
            "\n{} trace files to process= {}".format(
                len(files), "\n".join(map(str, files))
            )
        )

        # iterates over traces in folder
        for file in files:
            print(f"> Analyzing image {file}")
            save_folder = os.path.dirname(file)
            file_registered = shift_3d_mask(file)

            print(f"> will now run cellpose on {file_registered}")

            if cli:
                mask = run_cellpose(
                    file_registered,
                    diam,
                    cellprob,
                    flow,
                    stitch,
                    folder_destination=folder_mask_3d,
                    gpu=gpu,
                    pretrained_model=pretrained_model,
                )
            else:
                mask = run_cellpose_api(
                    file_registered,
                    diam,
                    cellprob,
                    flow,
                    stitch,
                    gpu=gpu,
                    pretrained_model=pretrained_model,
                )

            # Delete temp registered file
            os.unlink(file_registered)

            # Save mask in 3D
            new_mask_name = (
                save_folder
                + folder_mask_3d
                + os.sep
                + os.path.basename(file_registered).split(".")[0]
                + "_3Dmasks.npy"
            )
            if os.path.exists(new_mask_name):
                subprocess.run(f"mv {new_mask_name} {new_mask_name}_old", shell=True)
                print(
                    f"Warning: File already exists, we moved it to: {new_mask_name}_old"
                )
            print(f"> Saving output mask image to {new_mask_name}")
            np.save(new_mask_name, mask)

            # Save it also in 2D
            mask_2d_name = (
                save_folder
                + folder_mask_2d
                + os.sep
                + os.path.basename(file_registered).split(".")[0]
                + "_Masks.npy"
            )
            if os.path.exists(mask_2d_name):
                subprocess.run(f"mv {mask_2d_name} {mask_2d_name}_old", shell=True)
                print(
                    f"Warning: File already exists, we moved it to: {mask_2d_name}_old"
                )
            print(f"> Saving output mask image to {mask_2d_name}")
            np.save(mask_2d_name, np.max(mask, axis=0))


# =============================================================================
# MAIN
# =============================================================================


def main():
    # [parsing arguments]
    p = parseArguments()

    print("Remember to activate environment: conda activate cellpose!\n")

    if p["gpu"][0]:
        use_GPU = core.use_gpu()
        p["gpu"][1] = use_GPU
        print(">>> GPU activated? {}".format(use_GPU))
    else:
        print(">>> will use CPU")

    # [loops over lists of datafolders]
    process_images(
        cellprob=p["cellprob"],
        flow=p["flow"],
        stitch=p["stitch"],
        diam=p["diam"],
        files=p["files"],
        gpu=p["gpu"],
        cli=p["cli"],
        pretrained_model=p["model"],
    )

    print("Finished execution")


def find_roi_name_in_path(mask_3d_path):
    return os.path.basename(mask_3d_path).split("_")[3]


def find_label_in_path(mask_3d_path):
    return os.path.basename(mask_3d_path).split("_")[2]


def load_json(file_name):
    """Load a JSON file like a python dict

    Parameters
    ----------
    file_name : str
        JSON file name

    Returns
    -------
    dict
        Python dict
    """
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as json_file:
            return json.load(json_file)
    print(f"[WARNING] path {file_name} doesn't exist!")
    raise ValueError


def _shift_xy_mask_3d(image, shift):
    number_planes = image.shape[0]
    print(f"> Shifting {number_planes} planes")
    shift_3d = np.zeros((3))
    shift_3d[0], shift_3d[1], shift_3d[2] = 0, shift[0], shift[1]
    return shift_image(image, shift_3d)


def load_params():
    # loads dicShifts with shifts for all rois and all labels
    if os.path.exists("parameters.json"):
        print("Load parameters.json")
        return load_json("parameters.json")
    elif os.path.exists("infoList.json"):
        print(
            "[WARNING] 'infoList.json' is a DEPRECATED file name, please rename it 'parameters.json'"
        )
        print("Load infoList.json")
        return load_json("infoList.json")
    else:
        raise ValueError("[ERROR] 'parameters.json' file not found.")


def get_dict_shifts():
    params = load_params()
    dict_shifts_path = (
        params["common"]["alignImages"].get("register_global_folder", "register_global")
        + os.sep
        + "data"
        + os.sep
        + params["common"]["alignImages"].get("outputFile", "shifts")
        + ".json"
    )
    return load_json(dict_shifts_path)


def shift_3d_mask(mask_3d_path):

    roi_name = find_roi_name_in_path(mask_3d_path)
    label = find_label_in_path(mask_3d_path)
    dict_shifts = get_dict_shifts()

    # uses existing shift calculated by align_images
    try:
        shift = dict_shifts[f"ROI:{roi_name}"][label]
        print("> Applying existing XY shift...")
    except KeyError as e:
        shift = None
        raise SystemExit(
            f"# Could not find dictionary with alignment parameters for this ROI: ROI:{roi_name}, label: {label}"
        ) from e

    mask_3d = imread(mask_3d_path).squeeze()
    print(f"> image read: {mask_3d_path}")
    print(f"> Now registering image: {mask_3d_path}")

    # applies XY shift to 3D stack
    print(f"> Applies shift = [{shift[0]:.2f} ,{shift[1]:.2f}]")
    mask_3d_registered = _shift_xy_mask_3d(mask_3d, shift)

    # writes registered image to file
    mask_3d_registered_path = mask_3d_path.split(".")[0] + "_3d_registered.tif"
    imwrite(mask_3d_registered_path, mask_3d_registered)

    print(f"> wrote registered image to: {mask_3d_registered_path}")

    return mask_3d_registered_path


if __name__ == "__main__":
    main()
