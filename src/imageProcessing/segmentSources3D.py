#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:41:58 2021

@author: marcnol

Purpose: Localizes spots in 3D

steps:
    - load 3D file for cycle <i>
    - subtract background
    - load global alignment for this cycle
    - re-align 3D image using XY alignment
    - segment objects to get labeled masks
    - Get weighted moments and gaussian fits

    - display results:
        - overlap XY, XZ, YZ image projections with moment and gaussian fits
    - output results in a Table() using same formatting as that used in segmentMasks.py

"""

# =============================================================================
# IMPORTS
# =============================================================================

import glob
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from apifish.identification.spot_modeling import fit_subpixel
from apifish.image import projection
from astropy.table import Table, vstack
from scipy.stats import skew
from skimage import exposure, io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from traceratops.core.localization_table import create_output_table

from core.dask_cluster import try_get_client
from core.parameters import (
    AcquisitionParams,
    ProjectionParams,
    RegistrationParams,
    SegmentationParams,
    load_alignment_dict,
    print_dict,
)
from core.pyhim_logging import print_log, print_session_name, write_string_to_file
from core.saving import _plot_image_3d
from imageProcessing.alignImages import apply_xy_shift_3d_images
from imageProcessing.imageProcessing import image_adjust, preprocess_3d_image
from imageProcessing.makeProjections import reinterpolate_z
from imageProcessing.segmentMasks import (
    _deblend_3d_segmentation_advanced,
    _segment_3d_volumes_by_thresholding,
    _segment_3d_volumes_stardist,
)

# =============================================================================
# CLASSES
# =============================================================================


class Localize3D:
    def __init__(
        self,
        param,
        roi_name: str,
        acq_params: AcquisitionParams,
        proj_params: ProjectionParams,
        reg_params: RegistrationParams,
        seg_params: SegmentationParams,
        parallel=False,
    ):
        self.current_param = param
        self.parallel = parallel

        self.p = {}
        self.number_rois = None
        self.dict_shifts = None
        self.dict_shifts_available = None
        self.roi = roi_name
        self.filenames_to_process_list = []
        self.inner_parallel_loop = None
        self.output_filename = None
        self.single_file_to_process = None

        # parameters from parameters.json
        self.p["referenceBarcode"] = reg_params.referenceFiducial
        self.p["brightest"] = seg_params.brightest
        self.p["blockSizeXY"] = proj_params.block_size
        self.p["regExp"] = acq_params.fileNameRegExp
        self.p["zBinning"] = acq_params.zBinning

        self.p["zWindow"] = int(proj_params.zwindows / self.p["zBinning"])
        self.p["pixelSizeXY"] = acq_params.pixelSizeXY
        self.p["pixelSizeZ"] = acq_params.pixelSizeZ

        # decides what segmentation method to use
        self.p["3Dmethod"] = seg_params._3Dmethod
        self.p["reducePlanes"] = seg_params.reducePlanes

        # parameters used for 3D segmentation and deblending
        self.p["threshold_over_std"] = seg_params._3D_threshold_over_std
        self.p["sigma"] = seg_params._3D_sigma
        box_size = seg_params._3D_boxSize
        self.p["boxSize"] = (box_size, box_size)
        self.p["area_min"] = seg_params._3D_area_min
        self.p["area_max"] = seg_params._3D_area_max
        self.p["nlevels"] = seg_params._3D_nlevels
        self.p["contrast"] = seg_params._3D_contrast

        # parameters for stardist

        if seg_params.stardist_basename is not None and os.path.exists(
            seg_params.stardist_basename
        ):
            base_dir = seg_params.stardist_basename
        else:
            base_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                "stardist_models",
            )

        model_name = self._select_stardist_model(None, seg_params, base_dir)
        self.p["stardist_basename"] = base_dir
        self.p["stardist_network"] = model_name
        # parameters used for 3D gaussian fitting
        self.p["voxel_size_z"] = float(1000 * self.p["pixelSizeZ"] * self.p["zBinning"])
        self.p["voxel_size_yx"] = float(1000 * self.p["pixelSizeXY"])
        self.p["psf_z"] = seg_params._3D_psf_z
        self.p["psf_yx"] = seg_params._3D_psf_yx

        # range used for adjusting image levels during pre-precessing
        self.p["lower_threshold"] = seg_params._3D_lower_threshold
        self.p["higher_threshold"] = seg_params._3D_higher_threshold

        # parameters used for plotting 3D image
        # sets the number of planes around the center of the image used to represent localizations in XZ and ZY
        self.p["windowDisplay"] = 10

    def _select_stardist_model(
        self, label: Optional[str], seg_params: SegmentationParams, base_dir: str
    ) -> str:
        """Return the appropriate Stardist model name based on the label.

        If a custom network is configured and present on disk it is used.
        Otherwise, fall back to label-specific defaults.
        """

        if seg_params.stardist_network3D is not None:
            candidate_path = os.path.join(base_dir, seg_params.stardist_network3D)
            if os.path.exists(candidate_path):
                return seg_params.stardist_network3D

        if label == "DAPI":
            return "DAPI_3D_stardist_17032021_deconvolved"
        if label == "mask":
            return "PSF_3D_stardist_20210618_simu_deconvolved_thresh_0_01"

        return "PSF_3D_stardist_20210618_simu_deconvolved_thresh_0_01"

    def plot_image_3d(self, image_3d, localizations=None, masks=None, normalize_b=None):
        """
        makes list with XY, XZ and ZY projections and sends for plotting

        Parameters
        ----------
        image_3d : numpy array
            image in 3D.
        localizations : list
            list of localizations to overlay onto 3D image. The default is None.

        Returns
        -------
        figure handle

        """
        window = self.p["windowDisplay"]
        return _plot_image_3d(
            image_3d,
            localizations=localizations,
            masks=masks,
            normalize_b=normalize_b,
            window=window,
        )

    def _segment_3d_volumes(self, image_3d_aligned):
        p = self.p

        if p["3Dmethod"] == "stardist":
            binary = _segment_3d_volumes_stardist(
                image_3d_aligned,
                axis_norm=(0, 1, 2),
                model_dir=p["stardist_basename"],
                model_name=p["stardist_network"],
            )
            segmented_image_3d = binary
        else:
            binary, segmented_image_3d = _segment_3d_volumes_by_thresholding(
                image_3d_aligned,
                threshold_over_std=p["threshold_over_std"],
                sigma=p["sigma"],
                box_size=p["boxSize"],
                area_min=p["area_min"],
                area_max=p["area_max"],
                nlevels=p["nlevels"],
                contrast=p["contrast"],
                deblend_3d=True,
                parallel_execution=self.inner_parallel_loop,
            )

        return binary, segmented_image_3d

    def segment_sources_3d_file(self, filename_to_process, data_path, seg_params):
        p = self.p
        # excludes the reference fiducial and processes files in the same ROI
        roi = self.current_param.decode_file_parts(
            os.path.basename(filename_to_process)
        )["roi"]
        label = str(
            self.current_param.decode_file_parts(os.path.basename(filename_to_process))[
                "cycle"
            ]
        )

        self.p["stardist_network"] = self._select_stardist_model(
            label, seg_params, self.p["stardist_basename"]
        )

        # creates Table that will hold results
        output_table = create_output_table()

        # - load  and preprocesses 3D fiducial file
        print_log(f"\n\n>>>Processing roi:[{roi}] cycle:[{label}]<<<")
        print_log(f"$ File:{os.path.basename(filename_to_process)}")
        image_3d_0 = io.imread(filename_to_process).squeeze()

        # reinterpolates image in z if necessary
        image_3d_0 = reinterpolate_z(
            image_3d_0, range(0, image_3d_0.shape[0], p["zBinning"]), mode="remove"
        )

        # restricts analysis to a sub volume containing sources
        if p["reducePlanes"]:
            _, z_range, _ = projection.reinterpolate_focal_plane(
                image_3d_0, block_size_xy=p["blockSizeXY"], window=p["zWindow"]
            )
            z_correction = z_range[1][0]
            image_3d = image_3d_0[z_range[1], :, :].copy()
            print_log(
                f"$ Focal plane found: {z_range[0]}, z_range = {z_range[1]}, image_size = {image_3d.shape}"
            )
        else:
            image_3d = image_3d_0.copy()
            z_correction = 0
            print_log(f"$ z_range used = 0-{image_3d.shape[0]}")

        # preprocesses image by background subtraction and level normalization
        if p["3Dmethod"] != "stardist":
            image_3d = preprocess_3d_image(
                image_3d,
                p["lower_threshold"],
                p["higher_threshold"],
                parallel_execution=self.inner_parallel_loop,
            )

        # drifts 3D stack in XY or XYZ
        shift = None
        if self.dict_shifts_available and label != p["referenceBarcode"]:
            # uses existing shift calculated by align_images
            try:
                shift = self.dict_shifts[f"ROI:{roi}"][label]
                print_log("> Applying existing shift...")
            except KeyError:
                shift = None

        if shift is None and label != p["referenceBarcode"]:
            raise SystemExit(
                f"> Existing with ERROR: Could not find dictionary with alignment \
                    parameters for this ROI: ROI:{self.roi}, label: {label}"
            )

        # applies XY or XYZ shift to 3D stack
        if label != p["referenceBarcode"]:

            shift_arr = np.asarray(shift)
            if shift_arr.size >= 3:
                print_log(
                    f"$ Applies shift (z,x,y) = [{shift_arr[0]:.2f}, {shift_arr[1]:.2f}, {shift_arr[2]:.2f}]"
                )
            else:
                print_log(
                    f"$ Applies shift (x,y) = [{shift_arr[0]:.2f}, {shift_arr[1]:.2f}]"
                )
            image_3d_aligned = apply_xy_shift_3d_images(
                image_3d, shift, parallel_execution=self.inner_parallel_loop
            )
        else:
            print_log("$ Running reference fiducial cycle: no shift applied!")
            shift = np.array([0.0, 0.0])
            image_3d_aligned = image_3d

        processed_image_filename = (
            data_path
            + os.sep
            + seg_params.localize_3d_folder
            + os.sep
            + "data"
            + os.sep
            + os.path.splitext(os.path.basename(filename_to_process))[0]
            + "_processed.npy"
        )
        np.save(processed_image_filename, image_3d_aligned)
        print_log(f"$ Processed image saved to: {processed_image_filename}")

        # segments 3D volumes
        _, segmented_image_3d = self._segment_3d_volumes(image_3d_aligned)

        # deblend 3D volumes
        segmented_image_3d = _deblend_3d_segmentation_advanced(segmented_image_3d)

        # gets centroids and converts to spot int64 NPY array
        """
        (
            spots,
            sharpness,
            roundness1,
            roundness2,
            npix,
            sky,
            peak,
            flux,
            mag,
        ) = get_mask_properties(
            segmented_image_3d,
            image_3d_aligned,
            threshold=p["threshold_over_std"],
            n_tolerance=p["brightest"],
        )
        """

        (
            spots,
            snr_list,
            spot_pixel_percentage,
            mean_intensity_list,
            skew_list,
            patch_size,
            object_class,
            flux_list,
            roundness_list,
        ) = get_mask_properties_advanced(
            segmented_image_3d,
            image_3d_aligned,
        )

        number_sources = len(mean_intensity_list)
        print_log(
            f"$ Number of sources detected by image segmentation: {number_sources}"
        )

        if number_sources > 0:
            print_log("> Refits spots using gaussian 3D fittings...")

            print_log(" > Rescales image values after reinterpolation")
            image_3d_aligned = exposure.rescale_intensity(
                image_3d_aligned, out_range=(0, 1)
            )  # removes negative intensity levels

            # calls bigfish to get 3D sub-pixel coordinates based on 3D gaussian fitting
            try:
                # version 0.4
                spots_subpixel = fit_subpixel(
                    image_3d_aligned,
                    spots,
                    voxel_size_z=p["voxel_size_z"],
                    voxel_size_yx=p["voxel_size_yx"],
                    psf_z=p["psf_z"],
                    psf_yx=p["psf_yx"],
                )
            except TypeError:
                # version > 0.5
                spots_subpixel = fit_subpixel(
                    image_3d_aligned,
                    spots,
                    (
                        p["voxel_size_z"],
                        p["voxel_size_yx"],
                        p["voxel_size_yx"],
                    ),  # voxel size
                    (p["psf_z"], p["psf_yx"], p["psf_yx"]),
                )  # spot radius

            print_log(" > Updating table and saving results")

            # updates table
            for i in range(spots_subpixel.shape[0]):
                z, x, y = spots_subpixel[i, :]
                table_entry = [
                    str(uuid.uuid4()),
                    roi,
                    0,
                    int(label.split("RT")[1]),
                    i,
                    z + z_correction,
                    y,
                    x,
                    snr_list[i],
                    spot_pixel_percentage[i],
                    skew_list[i],
                    patch_size[i],
                    object_class[i],
                    mean_intensity_list[i],
                    flux_list[i],
                    roundness_list[i],
                ]
                output_table.add_row(table_entry)

            # represents image in 3D with localizations
            figures = []
            if p["3Dmethod"] == "stardist":
                image_3d_aligned = image_adjust(
                    image_3d_aligned,
                    lower_threshold=p["lower_threshold"],
                    higher_threshold=p["higher_threshold"],
                )[0]

            figures.append(
                [
                    self.plot_image_3d(
                        image_3d_aligned,
                        masks=segmented_image_3d,
                        localizations=[spots_subpixel, spots],
                    ),
                    "_3DimageNlocalizations.png",
                ]
            )

            # saves figures
            output_filenames = [
                data_path
                + os.sep
                + seg_params.localize_3d_folder
                + os.sep
                + os.path.basename(filename_to_process)
                + x[1]
                for x in figures
            ]

            for fig, file in zip(figures, output_filenames):
                fig[0].savefig(file)

        del image_3d_aligned, image_3d, image_3d_0

        return output_table

    def segment_sources_3d_in_folder(self, data_path, dict_shifts_path, seg_params):
        """
        Fits sources in all files in root_folder

        Returns
        -------
        None.

        """
        now = datetime.now()

        # Reads list of parameters assigned upon Class initialization
        p = self.p
        print_dict(p)

        # Finds images to process
        files_folder = glob.glob(data_path + os.sep + "*.tif")
        self.current_param.find_files_to_process(files_folder)
        nb_imgs = len(self.current_param.files_to_process)
        print_log(f"$ Number of images to be processed: {nb_imgs}")

        # loads dicShifts with shifts for all rois and all labels
        self.dict_shifts, self.dict_shifts_available = load_alignment_dict(
            dict_shifts_path
        )

        # creates Table that will hold results
        output_table_global = create_output_table()
        output_tables = []

        client = try_get_client()

        # loads reference fiducial image for this ROI
        self.filenames_to_process_list = [
            x
            for x in self.current_param.files_to_process
            if self.current_param.decode_file_parts(os.path.basename(x))["roi"]
            == self.roi
            and "RT"
            in self.current_param.decode_file_parts(os.path.basename(x))["cycle"]
        ]

        if self.single_file_to_process:
            self.filenames_to_process_list = [
                x
                for x in self.filenames_to_process_list
                if os.path.basename(x) == self.single_file_to_process
            ]
            if not self.filenames_to_process_list:
                raise SystemExit(
                    f"Requested file '{self.single_file_to_process}' was not found"
                    f" among the files to process in ROI [{self.roi}]."
                )

        n_files_to_process = len(self.filenames_to_process_list)
        print_log(f"$ Found {n_files_to_process} files in ROI [{self.roi}]")
        print_log(
            "$ [roi:cycle] {}".format(
                " | ".join(
                    [
                        str(
                            self.current_param.decode_file_parts(os.path.basename(x))[
                                "roi"
                            ]
                        )
                        + ":"
                        + str(
                            self.current_param.decode_file_parts(os.path.basename(x))[
                                "cycle"
                            ]
                        )
                        for x in self.filenames_to_process_list
                    ]
                )
            )
        )

        if client is None:
            self.inner_parallel_loop = True

            for file_index, filename_to_process in enumerate(
                self.filenames_to_process_list
            ):  # self.current_param.files_to_process):
                print_log(f"\n\n>>>Iteration: {file_index+1}/{n_files_to_process}<<<")

                output_tables.append(
                    self.segment_sources_3d_file(
                        filename_to_process, data_path, seg_params
                    )
                )
        else:
            self.inner_parallel_loop = False
            nb_workers = len(client.scheduler_info()["workers"])
            print_log(
                f"> Aligning {n_files_to_process} files using {nb_workers} workers..."
            )

            futures = [
                client.submit(self.segment_sources_3d_file, x, data_path, seg_params)
                for x in self.filenames_to_process_list
            ]

            output_tables = client.gather(futures)
            print_log(f" > Retrieving {len(output_tables)} results from cluster")

        # Merges Tables for different cycles and appends results Table to that of previous ROI
        output_table_global = vstack([output_table_global] + output_tables)

        print_log(f"$ localize_3d processing time: {datetime.now() - now}")

        # saves Table with all shifts in every iteration to avoid losing computed data
        output_table_global.write(
            self.output_filename,
            format="ascii.ecsv",
            overwrite=True,
        )

    def segment_sources_3d(
        self,
        data_path,
        dict_shifts_path,
        params: SegmentationParams,
        single_file_to_process: Optional[str] = None,
    ):
        """
        runs 3D fitting routine in root_folder

        Returns
        -------
        None.

        """
        session_name = "localize_3d"

        # processes folders and files

        print_session_name(session_name)
        write_string_to_file(
            self.current_param.param_dict["fileNameMD"],
            f"## {session_name}\n",
            "a",
        )

        # creates output folders and filenames
        self.single_file_to_process = single_file_to_process
        output_file_prefix = params.outputFile
        if self.single_file_to_process:
            base_name = os.path.splitext(os.path.basename(self.single_file_to_process))[
                0
            ]
            output_file_prefix = f"{output_file_prefix}_{base_name}"

        self.output_filename = (
            data_path
            + os.sep
            + params.localize_3d_folder
            + os.sep
            + "data"
            + os.sep
            + output_file_prefix
            + "_3D_barcode.ecsv"
        )

        print_log(f"> Processing Folder: {data_path}")

        self.segment_sources_3d_in_folder(data_path, dict_shifts_path, params)

        print_log(f"$ segmentedObjects run in {data_path} finished")

        return 0


##########################################
# FUNCTIONS
##########################################


def get_mask_properties(
    segmented_image_3d, image_3d_aligned, threshold=10, n_tolerance=1000
):
    """
    get object properties from labeled image and formats
    centroids in NPY array

    Parameters
    ----------
    segmented_image_3d : NPY 3D array
        labeled 3D image.
    image_3d_aligned : NPY 3D array
        pre-processed 3D image.

    Returns
    -------
    spots : NPY int64 array
        list of spots with the format: zyx

    """

    # gets object properties
    properties = regionprops(segmented_image_3d, intensity_image=image_3d_aligned)

    if len(properties) > 0:
        # selects n_tolerance brightest spots and keeps only these for further processing
        try:
            # compatibility with scikit_image versions <= 0.18
            peak0 = [x.max_intensity for x in properties]
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak0 = [x.intensity_max for x in properties]

        peak_list = peak0.copy()
        peak_list.sort()

        if n_tolerance == "None":
            last2keep = len(peak_list)
        else:
            last2keep = np.min([n_tolerance, len(peak_list)])

        highest_peak_value = peak_list[-last2keep]
        selection = list(np.nonzero(peak0 > highest_peak_value)[0])

        # attributes properties using the brightests spots selected
        try:
            # compatibility with scikit_image versions <= 0.18
            peak = [properties[x].max_intensity for x in selection]
            centroids = [properties[x].weighted_centroid for x in selection]
            sharpness = [
                float(properties[x].filled_area / properties[x].bbox_area)
                for x in selection
            ]
            roundness1 = [properties[x].equivalent_diameter for x in selection]
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak = [properties[x].intensity_max for x in selection]
            centroids = [properties[x].centroid_weighted for x in selection]
            sharpness = [
                float(properties[x].area_filled / properties[x].area_bbox)
                for x in selection
            ]
            roundness1 = [properties[x].equivalent_diameter_area for x in selection]

        roundness2 = [properties[x].extent for x in selection]
        npix = [properties[x].area for x in selection]
        sky = [0.0 for x in selection]

        try:
            # compatibility with scikit_image versions <= 0.18
            peak = [properties[x].max_intensity for x in selection]
            flux = [
                100 * properties[x].max_intensity / threshold for x in selection
            ]  # peak intensity over t$
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak = [properties[x].intensity_max for x in selection]
            flux = [
                100 * properties[x].intensity_max / threshold for x in selection
            ]  # peak intensity$

        mag = [-2.5 * np.log10(x) for x in flux]  # -2.5 log10(flux)

        # converts centroids to spot coordinates for bigfish to run 3D gaussian fits
        z = [x[0] for x in centroids]
        y = [x[1] for x in centroids]
        x = [x[2] for x in centroids]

        spots = np.zeros((len(z), 3))
        spots[:, 0] = z
        spots[:, 1] = y
        spots[:, 2] = x
        spots = spots.astype("int64")

        return (
            spots,
            sharpness,
            roundness1,
            roundness2,
            npix,
            sky,
            peak,
            flux,
            mag,
        )
    else:
        # creates output lists to return
        return [], [], [], [], [], [], [], [], []


def _safe_region_axis_lengths(region):
    """Return finite 3D region axis lengths, falling back for degenerate objects.

    scikit-image derives axis lengths from inertia-tensor eigenvalues. Very
    small, flat, or numerically degenerate 3D labels can produce a slightly
    negative value inside the square-root used for ``axis_minor_length`` or
    ``axis_major_length``, which raises ``ValueError: math domain error``.
    Treat those labels as valid detections but mark their unavailable axis
    length as 0 so downstream roundness/elongation metrics remain safe.
    """

    try:
        minor_axis = region.minor_axis_length
    except ValueError:
        print_log(
            f"$ Warning: could not compute minor axis length for label "
            f"{region.label}; using 0. This usually indicates a degenerate "
            "or very small 3D object."
        )
        minor_axis = 0

    try:
        major_axis = region.major_axis_length
    except ValueError:
        print_log(
            f"$ Warning: could not compute major axis length for label "
            f"{region.label}; using 0. This usually indicates a degenerate "
            "or very small 3D object."
        )
        major_axis = 0

    if not np.isfinite(minor_axis) or minor_axis < 0:
        minor_axis = 0
    if not np.isfinite(major_axis) or major_axis < 0:
        major_axis = 0

    return minor_axis, major_axis


def get_mask_properties_advanced(segmented_image_3d, image_3d_aligned):
    """
    Extract shape and position features from 3D labeled objects.

    Parameters
    ----------
    segmented_image_3d : ndarray
        Labeled 3D segmentation.
    image_3d_aligned : ndarray
        Intensity image.

    Returns
    -------
    astropy.table.Table
        One row per object with centroid, bbox, area, and shape metrics.
    """

    properties = regionprops(segmented_image_3d, intensity_image=image_3d_aligned)

    if len(properties) == 0:
        return Table(
            names=[
                "z",
                "y",
                "x",
                "z_min",
                "y_min",
                "x_min",
                "z_max",
                "y_max",
                "x_max",
                "area",
                "minor_axis_length",
                "major_axis_length",
                "elongation",
            ],
            rows=[],
        )

    try:
        centroids = [p.weighted_centroid for p in properties]
    except AttributeError:
        centroids = [p.centroid_weighted for p in properties]

    bbox = [p.bbox for p in properties]
    area = [p.area for p in properties]
    minor_axis_length = []
    major_axis_length = []

    for prop in properties:
        minor_axis, major_axis = _safe_region_axis_lengths(prop)
        minor_axis_length.append(minor_axis)
        major_axis_length.append(major_axis)

    elongation = [
        major_axis_length[i] / minor_axis_length[i] if minor_axis_length[i] > 0 else 0
        for i in range(len(properties))
    ]

    mask_properties = Table(
        {
            "z": [c[0] for c in centroids],
            "y": [c[1] for c in centroids],
            "x": [c[2] for c in centroids],
            "z_min": [b[0] for b in bbox],
            "y_min": [b[1] for b in bbox],
            "x_min": [b[2] for b in bbox],
            "z_max": [b[3] for b in bbox],
            "y_max": [b[4] for b in bbox],
            "x_max": [b[5] for b in bbox],
            "area": area,
            "minor_axis_length": minor_axis_length,
            "major_axis_length": major_axis_length,
            "elongation": elongation,
        }
    )

    return spot_quality_metrics(image_3d_aligned, mask_properties)


def spot_quality_metrics(image_3d_aligned, mask_properties):
    """
    Compute intensity and quality metrics for each segmented spot.

    Metrics include:
    - SNR
    - Mean spot intensity
    - Spot pixel fraction
    - Patch skewness
    - Patch size
    - Object classification (spot/background)

    Parameters
    ----------
    image_3d_aligned : ndarray or str
        3D image or path to numpy file.
    mask_properties : Table or DataFrame
        Spot properties (bbox, centroid).

    Returns
    -------
    tuple
            (spots,
            snr_list,
            spot_pixel_percentage,
            mean_intensity_list,
            skew_list,
            patch_size,
            type_object,
            flux,
            roundness)
    """

    mask_properties = mask_properties.to_pandas()

    snr_list = []
    spot_pixel_percentage = []
    mean_intensity_list = []
    object_count_list = []
    skew_list = []
    patch_size = []
    xmin, xmax = [], []
    ymin, ymax = [], []
    zmin, zmax = [], []
    roundness = []
    spot_x, spot_y, spot_z = [], [], []
    number_skipped_spots_window_size = 0
    number_skipped_spots_centroid_nan = 0

    xy_min_window_size = 5  # minimum size of the patch to compute metrics in xy
    z_min_window_size = 5  # minimum size of the patch to compute metrics in z

    if len(mask_properties) > 0:

        for idx, row in mask_properties.iterrows():

            # Extract the local 3D patch using the bounding box coordinates
            y_min, y_max = round(row["y_min"]), round(row["y_max"])
            x_min, x_max = round(row["x_min"]), round(row["x_max"])
            z_min, z_max = round(row["z_min"]), round(row["z_max"])

            # does not calculate metrices for patches that are too small in XY
            window_size = np.array([(z_max - z_min), (y_max - y_min), (x_max - x_min)])

            if (
                np.any(window_size[1:] < xy_min_window_size)
                or window_size[0] < z_min_window_size
                or np.any(np.isnan([row["x"], row["y"], row["z"]]))
            ):
                print_log(
                    f"$ Skipping spot at (z,y,x)=({row['z']:.2f},{row['y']:.2f},{row['x']:.2f}) "
                    f"with window size (z,y,x)=({window_size[0]},{window_size[1]},{window_size[2]})"
                )
                number_skipped_spots_window_size += 1
                continue

            x_int = int(round(row["x"]))
            y_int = int(round(row["y"]))
            z_int = int(round(row["z"]))

            if np.any(
                [
                    x_int < x_min,
                    x_int >= x_max,
                    y_int < y_min,
                    y_int >= y_max,
                    z_int < z_min,
                    z_int >= z_max,
                ]
            ):
                number_skipped_spots_centroid_nan += 1
                continue

            # If we reach this point, the spot is valid and we can compute its metrics

            # Convert global coordinates into local patch coordinates
            z_local = z_int - z_min
            y_local = y_int - y_min
            x_local = x_int - x_min

            # will save the spots that passed the tests above only, the others are removed from the lists
            xmin.append(x_min)
            xmax.append(x_max)
            ymin.append(y_min)
            ymax.append(y_max)
            zmin.append(z_min)
            zmax.append(z_max)
            spot_x.append(x_int)
            spot_y.append(y_int)
            spot_z.append(z_int)

            roundness.append(row["elongation"])

            patch = image_3d_aligned[z_min:z_max, y_min:y_max, x_min:x_max]
            patch_size.append(round((patch.size)))

            # Segment the patch using Otsu thresholding
            threshold = threshold_otsu(patch)
            mask_spot = patch > threshold

            # Compute signal-to-noise (SNR) ratio using the local background
            background = patch[patch <= threshold]

            if background.size == 0 or np.std(background) == 0:
                snr_list.append(np.nan)
            else:
                I_spot = image_3d_aligned[z_int, y_int, x_int]
                snr = (I_spot - np.mean(background)) / np.std(background)
                snr_list.append(snr)

            # Identify connected components inside the segmented patch
            labeled_mask = label(mask_spot)
            regions = regionprops(labeled_mask)
            object_count_list.append(len(regions))

            # Check if the spot coordinates are within the bounds of the labeled mask
            label_at_point = labeled_mask[z_local, y_local, x_local]

            if label_at_point == 0:
                spot_pixel_percentage.append(np.nan)
                mean_intensity_list.append(np.nan)
            else:
                region = next(r for r in regions if r.label == label_at_point)

                # Compute spot size and mean intensity
                spot_pixel_percentage.append(region.area / patch.size * 100)

                mean_intensity_list.append(
                    round(np.mean(patch[labeled_mask == region.label]))
                )

            # Compute intensity distribution asymmetry
            skew_list.append(skew(patch.ravel()))

        print(
            f"Number of skipped spots (window size): {number_skipped_spots_window_size}"
        )
        print(
            f"Number of skipped spots (centroid NaN): {number_skipped_spots_centroid_nan}"
        )

        # Classify each patch as a spot or background
        type_object = [1 if not np.isnan(v) else 0 for v in spot_pixel_percentage]

        # assigns spot coordinates to a numpy array for downstream processing
        spots = np.zeros((len(spot_z), 3), dtype=np.int64)
        spots[:, 0] = np.array(spot_z).round().astype(int)
        spots[:, 1] = np.array(spot_y).round().astype(int)
        spots[:, 2] = np.array(spot_x).round().astype(int)

        # spots = np.zeros((len(mask_properties), 3), dtype=np.int64)

        # spots[:, 0] = mask_properties["z"].round().astype(int)
        # spots[:, 1] = mask_properties["y"].round().astype(int)
        # spots[:, 2] = mask_properties["x"].round().astype(int)

        flux = snr_list

        return (
            spots,
            snr_list,
            spot_pixel_percentage,
            mean_intensity_list,
            skew_list,
            patch_size,
            type_object,
            flux,
            roundness,
        )

    else:
        return [], [], [], [], [], [], [], [], []
