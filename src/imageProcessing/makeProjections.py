#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:17:58 2020

@author: marcnol

This file contains functions to project 3D images to 2D

Operation will be defined in the parameters file. Options are:
    - user-defined range
    - all z range
    - optimal range based on detection of focal plane and use of user defined window around it


"""
# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np
import scipy.optimize as spo
from apifish.stack import projection

from core.parameters import Parameters
from core.pyhim_logging import print_log

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")


class Feature:
    def __init__(self, params: Parameters):
        self.m_params = params
        self.required_data = []
        self.required_ref = []
        self.required_table = []
        self.out_folder = ""

    def get_required_inputs(self):
        return self.required_data, self.required_ref, self.required_table


class Project(Feature):
    def __init__(self, params: Parameters):
        super().__init__(params)
        self.required_data = ["barcode", "mask", "dapi", "fiducial", "rna"]
        self.out_folder = params.param_dict["common"]["zProject"]["folder"]

        self.block_size = params.get_labeled_dict_value("zProject", "blockSize")
        self.display = params.get_labeled_dict_value("zProject", "display")
        self.mode = params.get_labeled_dict_value("zProject", "mode")
        self.operation = params.get_labeled_dict_value("zProject", "operation")
        self.save_image = params.get_labeled_dict_value("zProject", "saveImage")
        self.window_security = params.get_labeled_dict_value(
            "zProject", "windowSecurity"
        )
        self.z_project_option = params.get_labeled_dict_value(
            "zProject", "zProjectOption"
        )
        self.zmax = params.get_labeled_dict_value("zProject", "zmax")
        self.zmin = params.get_labeled_dict_value("zProject", "zmin")
        self.zwindows = params.get_labeled_dict_value("zProject", "zwindows")

    def find_out_tags(self, label):
        tags = ["_2d"]
        # TODO: Check if label exist
        # (may be this test have to be done in high level like Pipeline initialization)
        if self.mode[label] == "laplacian":
            tags.append("_focalPlaneMatrix")
        return tags

    def run(self, img, label: str):
        mode = self.mode[label]
        if mode == "laplacian":
            return self._projection_laplacian(img, label)
        # find the correct range for the projection
        img_reduce = self.precise_z_planes(img, mode, label)
        img_projected = self.projection_2d(img_reduce, label)
        return [img_projected]

    def check_zmax(self, img_size, label):
        if self.zmax[label] > img_size[0]:
            print_log("$ Setting z max to the last plane")
            self.zmax[label] = img_size[0]

    def precise_z_planes(self, img, mode, label):
        img_size = img.shape
        self.check_zmax(img_size, label)
        if mode == "automatic":
            focus_plane, z_range = self._precise_z_planes_auto(img, label)
        elif mode == "full":
            focus_plane, z_range = self._precise_z_planes_full(img_size)
        elif mode == "manual":
            focus_plane, z_range = self._precise_z_planes_manual(label)
        else:
            raise ValueError(
                f"Projection mode UNRECOGNIZED: {mode}\n> Available mode: automatic,full,manual,laplacian"
            )
        self.__print_img_properties(z_range, img_size, focus_plane)
        return img[z_range[0] : z_range[-1] + 1]

    def _precise_z_planes_auto(self, img, label):
        """
        Calculates the focal planes based max standard deviation
        Finds best focal plane by determining the max of the std deviation vs z curve
        """
        win_sec = self.window_security[label]

        print_log("> Calculating planes...")

        nb_of_planes = self.zmax[label] - self.zmin[label]
        std_matrix = np.zeros(nb_of_planes)
        mean_matrix = np.zeros(nb_of_planes)

        # calculate STD in each plane
        for i in range(nb_of_planes):
            std_matrix[i] = np.std(img[i])
            mean_matrix[i] = np.mean(img[i])

        max_std = np.max(std_matrix)
        i_focus_plane = np.where(std_matrix == max_std)[0][0]
        # Select a window to avoid being on the edges of the stack

        if i_focus_plane < win_sec or (i_focus_plane > nb_of_planes - win_sec):
            focus_plane = i_focus_plane
        else:
            # interpolate zfocus
            axis_z = range(
                max(
                    self.zmin[label],
                    i_focus_plane - win_sec,
                    min(self.zmax[label], i_focus_plane + win_sec),
                )
            )
            std_matrix -= np.min(std_matrix)
            std_matrix /= np.max(std_matrix)

            try:
                fitgauss = spo.curve_fit(
                    projection.gaussian, axis_z, std_matrix[axis_z[0] : axis_z[-1] + 1]
                )
                focus_plane = int(fitgauss[0][1])
            except RuntimeError:
                print_log("Warning, too many iterations", status="WARN")
                focus_plane = i_focus_plane
        zmin = max(win_sec, focus_plane - self.zwindows[label])
        zmax = min(
            nb_of_planes, win_sec + nb_of_planes, focus_plane + self.zwindows[label]
        )
        zrange = range(zmin, zmax + 1)

        return focus_plane, zrange

    def _precise_z_planes_full(self, img_size):
        (zmin, zmax) = (0, img_size[0])
        focus_plane = round((zmin + zmax) / 2)
        z_range = range(zmin, zmax)
        return focus_plane, z_range

    def _precise_z_planes_manual(self, label):
        # Manual: reads from parameters file
        if self.zmin[label] >= self.zmax[label]:
            raise SystemExit(
                "zmin is equal or larger than zmax in configuration file. Cannot proceed."
            )
        focus_plane = round((self.zmin[label] + self.zmax[label]) / 2)
        z_range = range(self.zmin[label], self.zmax[label])
        return focus_plane, z_range

    def _projection_laplacian(self, img, label):
        print_log("Stacking using Laplacian variance...")
        focal_plane_matrix, z_range, block = projection.reinterpolate_focal_plane(
            img, block_size_xy=self.block_size[label], window=self.zwindows[label]
        )
        self.__print_img_properties(z_range[1], img.shape, z_range[0])
        # reassembles image
        output = projection.reassemble_images(
            focal_plane_matrix, block, window=self.zwindows[label]
        )

        return output, (focal_plane_matrix, z_range[0])

    def projection_2d(self, img, label):
        # sums images
        i_collapsed = None
        option = self.z_project_option[label]
        if option == "MIP":
            # Max projection of selected planes
            i_collapsed = projection.maximum_projection(img)
            # i_collapsed = projection.maximum_projection(img[:-1])
        elif option == "sum":
            # Sums selected planes
            i_collapsed = projection.sum_projection(img)
        else:
            print_log(
                f"ERROR: mode not recognized. Expected: MIP or sum. Read: {option}"
            )

        return i_collapsed

    @staticmethod
    def __print_img_properties(z_range, size, focal_plane):
        # Outputs image properties to command line
        print_log(f"> Processing z_range:{z_range}")
        print_log(f"$ Image Size={size}")
        print_log(f"$ Focal plane={focal_plane}")


# =============================================================================
# FOCAL PLANE INTERPOLATION
# =============================================================================


def _remove_z_planes(image_3d, z_range):
    """
    Removes planes in input image.
    For instance, if you provide a z_range = range(0,image_3d.shape[0],2)

    then the routine will remove any other plane. Number of planes skipped
    can be programmed by tuning z_range.

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.
    mode : str, optional
        'remove' will remove planes
        'interpolate' will perform an interpolation
        The default is 'remove'.

    Returns
    -------
    output: numpy array

    """
    output = np.zeros((len(z_range), image_3d.shape[1], image_3d.shape[2]))
    for i, index in enumerate(z_range):
        output[i, :, :] = image_3d[index, :, :]

    return output


def _average_z_planes(image_3d, z_range):
    """
    Removes z-planes by calculating the average between successive planes

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.
    mode : str, optional
        'remove' will remove planes
        'interpolate' will perform an interpolation
        The default is 'remove'.

    Returns
    -------
    output: numpy array

    """

    output = np.zeros((len(z_range), image_3d.shape[1], image_3d.shape[2]))
    for i, index in enumerate(z_range):
        average = (
            image_3d[index, :, :].astype(np.float)
            + image_3d[index + 1, :, :].astype(np.float)
        ) / 2
        output[i, :, :] = average.astype(np.uint16)

    return output


def _interpolate_z_planes(image_3d, z_range):
    """
    Removes z planes by reinterpolation

    TODO

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.
    mode : str, optional
        'remove' will remove planes
        'interpolate' will perform an interpolation
        The default is 'remove'.

    Returns
    -------
    output: numpy array

    """

    output = np.zeros((len(z_range), image_3d.shape[1], image_3d.shape[2]))

    # need to code using interpn
    output = image_3d

    return output


def reinterpolate_z(image_3d, z_range, mode="average"):
    """
    wrapper function for any kind of z-interpolation
    to reduce the number of planes in an image

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    z_range : range
        range of planes for the output image.
    mode : str, optional
        'remove' will remove planes
        'interpolate' will perform an interpolation
        The default is 'remove'.

    Returns
    -------
    output: numpy array

    """
    if mode == "interpolate":
        output = _interpolate_z_planes(image_3d, z_range)
    elif mode == "remove":
        output = _remove_z_planes(image_3d, z_range)
    elif mode == "average":
        output = _average_z_planes(image_3d, z_range)

    print_log(f"$ Reduced Z-planes from {image_3d.shape[0]} to {output.shape[0]}")

    return output
