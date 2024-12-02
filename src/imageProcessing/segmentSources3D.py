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

import numpy as np
from apifish.detection.spot_modeling import fit_subpixel
from apifish.stack import projection
from astropy.table import Table, vstack
from skimage import exposure, io
from skimage.measure import regionprops

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
        if seg_params.stardist_network3D is not None and os.path.exists(
            os.path.join(base_dir, seg_params.stardist_network3D)
        ):
            model_name = seg_params.stardist_network3D
        else:
            model_name = "PSF_3D_stardist_20210618_simu_deconvolved_thresh_0_01"
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
            binary, segmented_image_3d = _segment_3d_volumes_stardist(
                image_3d_aligned,
                deblend_3d=True,
                axis_norm=(0, 1, 2),
                model_dir=p["stardist_basename"],
                model_name=p["stardist_network"],
            )
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
            z_offset = z_range[1][0]
            image_3d = image_3d_0[z_range[1], :, :].copy()
            print_log(
                f"$ Focal plane found: {z_range[0]}, z_range = {z_range[1]}, image_size = {image_3d.shape}"
            )
        else:
            image_3d = image_3d_0.copy()
            z_offset = 0
            print_log(f"$ z_range used = 0-{image_3d.shape[0]}")

        # preprocesses image by background subtraction and level normalization
        if p["3Dmethod"] != "stardist":
            image_3d = preprocess_3d_image(
                image_3d,
                p["lower_threshold"],
                p["higher_threshold"],
                parallel_execution=self.inner_parallel_loop,
            )

        # drifts 3D stack in XY
        shift = None
        if self.dict_shifts_available and label != p["referenceBarcode"]:
            # uses existing shift calculated by align_images
            try:
                shift = self.dict_shifts[f"ROI:{roi}"][label]
                print_log("> Applying existing XY shift...")
            except KeyError:
                shift = None

        if shift is None and label != p["referenceBarcode"]:
            raise SystemExit(
                f"> Existing with ERROR: Could not find dictionary with alignment \
                    parameters for this ROI: ROI:{self.roi}, label: {label}"
            )

        # applies XY shift to 3D stack
        if label != p["referenceBarcode"]:
            print_log(f"$ Applies shift = [{shift[0]:.2f} ,{shift[1]:.2f}]")
            image_3d_aligned = apply_xy_shift_3d_images(
                image_3d, shift, parallel_execution=self.inner_parallel_loop
            )
        else:
            print_log("$ Running reference fiducial cycle: no shift applied!")
            shift = np.array([0.0, 0.0])
            image_3d_aligned = image_3d
        # segments 3D volumes
        _, segmented_image_3d = self._segment_3d_volumes(image_3d_aligned)

        # gets centroids and converts to spot int64 NPY array
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

        number_sources = len(peak)
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
            # compatibility with latest version of bigfish. To be removed if stable.
            # TODO: Is it stable ? I think we can remove it.
            try:
                # version 0.4 commit fa0df4f
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
                    z + z_offset,
                    y,
                    x,
                    sharpness[i],
                    roundness1[i],
                    roundness2[i],
                    npix[i],
                    sky[i],
                    peak[i],
                    flux[i],
                    mag[i],
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
        self, data_path, dict_shifts_path, params: SegmentationParams
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
        self.output_filename = (
            data_path
            + os.sep
            + params.localize_3d_folder
            + os.sep
            + "data"
            + os.sep
            + params.outputFile
            + "_3D_barcode.dat"
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


def create_output_table():
    output = Table(
        names=(
            "Buid",
            "ROI #",
            "CellID #",
            "Barcode #",
            "id",
            "zcentroid",
            "xcentroid",
            "ycentroid",
            "sharpness",
            "roundness1",
            "roundness2",
            "npix",
            "sky",
            "peak",
            "flux",
            "mag",
        ),
        dtype=(
            "S2",
            "int",
            "int",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )
    return output
