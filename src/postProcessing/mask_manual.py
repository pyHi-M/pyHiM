#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:47:13 2023

@author: marcnol
"""
import argparse
import json
import os
import sys

import numpy as np
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from roipoly import MultiRoi
from scipy.ndimage import shift as shift_image
from skimage import io
from tifffile import imread, imwrite


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

    # applies XY shift to 3D stack
    print(f"$ Applies shift = [{shift[0]:.2f} ,{shift[1]:.2f}]")
    mask_3d_registered = _shift_xy_mask_3d(mask_3d, shift)
    mask_3d_registered_path = mask_3d_path.split(".")[0] + "_3d_registered.tif"
    imwrite(mask_3d_registered_path, mask_3d_registered)
    return mask_3d_registered_path


def show_image(data_2d, normalization="simple", size=(10, 10)):
    fig = plt.figure()
    fig.set_size_inches(size)

    axes = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()

    if normalization == "simple":
        norm = simple_norm(data_2d, "sqrt", percent=99.9)

    fig.add_axes(axes)
    axes.imshow(data_2d, origin="lower", cmap="Greys_r", norm=norm)
    axes.set_title("2D Data")
    return fig, axes


def creates_user_mask(file_name, label):
    # loads image

    # displays image
    data = io.imread(file_name).squeeze()
    print(f"$ Image size: {data.shape}")
    data_2d = np.max(data, axis=0)

    fig = plt.figure()
    fig.set_size_inches((10, 10))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)
    ax.set_axis_off()
    norm = simple_norm(data_2d, "sqrt", percent=99.9)
    ax.imshow(data_2d, origin="lower", cmap="Greys_r", norm=norm)

    print("Click on the button to add a new ROI")

    # Draw multiple rois
    multiroi_named = MultiRoi(roi_names=["ROI1", "ROI2"])

    number_rois = len(multiroi_named.rois)
    print(f"Number of rois drawn: {number_rois}")

    masks = np.zeros((data_2d.shape[0], data_2d.shape[1], number_rois))

    # Display image with all rois
    fig, _ = show_image(data_2d)

    roi_names = []
    i = 0
    for name, roi in multiroi_named.rois.items():
        roi.display_roi()
        roi.display_mean(data_2d)
        roi_names.append(name)
        masks[:, :, i] = roi.get_mask(data_2d)
        i += 1
    plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
    # plt.show()

    print("Saving and closing image with rois...")
    fig.savefig(file_name.split(".")[0] + "_" + label + ".png")
    plt.close(fig)

    # saves result
    output = os.path.basename(file_name).split(".")[0] + "_" + label + ".npy"
    np.save(output, masks)
    print(f"$ saved output image mask as {output}")


def main():

    print("=" * 10 + "Started execution" + "=" * 10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file name (TIF assumed)")
    parser.add_argument("--label", help="Label to add to image file name. Default=none")
    args = parser.parse_args()

    if args.input:
        input_file = args.input
    else:
        print(">> ERROR: you must provide a filename with the image file")
        sys.exit(-1)

    if args.label:
        label = args.label
    else:
        label = ""

    print(f"parameters> input_file: {input_file}\n")

    file_registered = shift_3d_mask(input_file)
    creates_user_mask(file_registered, label)
    # Delete tempo registered file
    os.unlink(file_registered)

    print("=" * 9 + "Finished execution" + "=" * 9)


if __name__ == "__main__":
    main()
