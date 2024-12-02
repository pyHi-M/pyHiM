#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:22:18 2020

@author: marcnol

Sets of functions that do alignment of 2D fiducial images. It also contains
code to apply these alignments to other channels (masks/ barcodes)

For the time being alignment is purely based on optimized sub-pixed accuracy
image cross correlation

"""

# =============================================================================
# IMPORTS
# =============================================================================

import glob
import os
import sys

import numpy as np
from astropy.stats import SigmaClip
from astropy.table import Table
from numpy import linalg as LA
from photutils import Background2D, MedianBackground
from scipy.ndimage import shift as shift_image
from skimage import exposure, measure
from skimage.exposure import match_histograms
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from skimage.util.shape import view_as_blocks
from tqdm import tqdm, trange

from core.dask_cluster import try_get_client
from core.data_file import (
    BlockAlignmentFile,
    BothImgRbgFile,
    EcsvFile,
    EqualizationHistogramsFile,
    JsonFile,
    NpyFile,
    RefDiffFile,
)
from core.data_manager import load_json
from core.parameters import ProjectionParams, RegistrationParams
from core.pyhim_logging import print_log
from imageProcessing.imageProcessing import (
    Image,
    image_adjust,
    reassemble_3d_image,
    scatter_3d_image,
)
from imageProcessing.makeProjections import Feature


def preprocess_2d_img(img, background_sigma):
    # Normalises images
    norm_img = img / img.max()
    # removes inhomogeneous background
    return remove_inhomogeneous_background(norm_img, background_sigma)


class RegisterGlobal(Feature):
    def __init__(self, params: RegistrationParams):
        super().__init__(params)
        self.npy_labels = ["fiducial"]
        self.required_ref = {
            "data_type": "npy",
            "label_part": params.referenceFiducial,
            "label": "fiducial",
        }
        self.out_folder = self.params.register_global_folder
        self.name = "RegisterGlobal"

    def run(self, raw_2d_img, reference_2d_img):
        if np.array_equal(raw_2d_img, reference_2d_img, equal_nan=True):
            return [NpyFile(reference_2d_img, "_2d_registered")], None
        results_to_save = []
        preprocessed_img = preprocess_2d_img(raw_2d_img, self.params.background_sigma)
        preprocessed_ref = preprocess_2d_img(
            reference_2d_img, self.params.background_sigma
        )

        if self.params.alignByBlock:
            (
                preprocessed_ref,
                preprocessed_img,
                shift,
                diffphase,
                relative_shifts,
                rms_image,
                contour,
            ) = compute_shift_by_block(
                preprocessed_ref,
                preprocessed_img,
                self.params.blockSize,
                self.params.tolerance,
            )

            results_to_save.append(
                BlockAlignmentFile(relative_shifts, rms_image, contour)
            )

            # saves mask of valid regions with a correction within the tolerance
            results_to_save.append(NpyFile(rms_image, "_rmsBlockMap"))
            results_to_save.append(NpyFile(relative_shifts, "_errorAlignmentBlockMap"))
        else:
            shift, diffphase, i_histogram, lower_threshold = compute_global_shift(
                preprocessed_ref,
                preprocessed_img,
                self.params.lower_threshold,
                self.params.higher_threshold,
            )

            results_to_save.append(
                EqualizationHistogramsFile(i_histogram, lower_threshold)
            )

        shifted_img = shift_image(preprocessed_img, shift)
        error = calcul_error(shifted_img, preprocessed_ref)
        # thresholds corrected images for better display and saves
        preprocessed_ref[preprocessed_ref < 0] = 0
        preprocessed_img[preprocessed_img < 0] = 0
        results_to_save.append(BothImgRbgFile(preprocessed_ref, shifted_img))
        results_to_save.append(
            RefDiffFile(preprocessed_ref, shifted_img, preprocessed_img)
        )
        results_to_save.append(NpyFile(shifted_img, "_2d_registered"))

        results_to_keep = {"shift": shift, "diffphase": diffphase, "error": error}
        return results_to_save, results_to_keep

    def merge_results(self, results: list[dict]):
        dict_shift_roi = {}
        alignment_results_table = Table(
            names=(
                "aligned file",
                "reference file",
                "shift_x",
                "shift_y",
                "error",
                "diffphase",
            ),
            dtype=("S2", "S2", "f4", "f4", "f4", "f4"),
        )
        for result_dict in results:
            label_part = result_dict["cycle"]
            shift = result_dict["shift"]
            table_entry = [
                result_dict["tif_name"],
                result_dict["ref_tif_name"],
                result_dict["shift"][0],
                result_dict["shift"][1],
                result_dict["error"],
                result_dict["diffphase"],
            ]
            dict_shift_roi[label_part] = shift.tolist()
            alignment_results_table.add_row(table_entry)

        roi = results[0]["roi"]
        dict_shifts = {f"ROI:{roi}": dict_shift_roi}

        return [JsonFile(dict_shifts), EcsvFile(alignment_results_table)]


class ApplyRegisterGlobal(Feature):
    def __init__(self, params: RegistrationParams):
        super().__init__(params)
        # self.required_data = ["barcode", "mask", "DAPI", "RNA"]
        # self.required_ref = params.referenceFiducial
        # self.required_table = ["shift"]
        self.out_folder = self.params.register_global_folder
        self.name = "ApplyRegisterGlobal"

    # def run(self, raw_2d_img, dict_shifts:dict, raw_label:str="RT42", roi_name:str = "001"):
    #      """
    #     Applies registration

    #     """
    #     if raw_label == self.params.referenceFiducial:
    #         return raw_2d_img,"_2d_registered"
    #     try:
    #         # gets shift from dictionary
    #         shift_array = dict_shifts[f"ROI:{roi_name}"][raw_label]
    #     except KeyError:
    #         shift_array = None
    #         msg = f"$ Could not find dictionary with alignment parameters for this ROI: ROI:{roi_name}, label: {raw_label}"
    #         print_log(msg)
    #         raise KeyError(msg)

    #     shift = np.asarray(shift_array)
    #     registered_2d_img = shift_image(raw_2d_img, shift)
    #     print_log(f"$ Image registered using ROI:{roi_name}, label:{raw_label}, shift={shift}")

    #     return registered_2d_img,"_2d_registered"


#      ||
#      ||
#      ||
#      ||
#      ||
#      ||
#      ||
#      ||
# No-refactored
# \            /
# \          /
#  \        /
#   \      /
#    \    /
#     \  /
#      \/

np.seterr(divide="ignore", invalid="ignore")

# =============================================================================
# FUNCTIONS
# =============================================================================


def remove_inhomogeneous_background(im, background_sigma):
    sigma_clip = SigmaClip(sigma=background_sigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        im,
        (64, 64),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    return im - bkg.background


def align_images_by_blocks(
    img_1,
    img_2,
    block_size,
    upsample_factor=100,
    min_number_pollsters=4,
    tolerance=0.1,
    shift_error_tolerance=5,
):
    block_1 = view_as_blocks(img_1, block_size)
    block_2 = view_as_blocks(img_2, block_size)
    shift_image_norm = np.zeros((block_1.shape[0], block_1.shape[1]))
    shifted_image = np.zeros((block_1.shape[0], block_1.shape[1], 2))
    rms_image = np.zeros((block_1.shape[0], block_1.shape[1]))

    for i in trange(block_1.shape[0]):
        for j in range(block_1.shape[1]):
            # using Scimage registration functions
            shift, _, _ = phase_cross_correlation(
                block_1[i, j], block_2[i, j], upsample_factor=upsample_factor
            )
            shift_image_norm[i, j] = LA.norm(shift)
            shifted_image[i, j, 0], shifted_image[i, j, 1] = shift[0], shift[1]
            img_2_aligned = shift_image(img_2, shift)

            rms_image[i, j] = np.sum(np.sum(np.abs(img_1 - img_2_aligned), axis=1))

    # [calculates optimal shifts by polling blocks showing the best RMS]

    # threshold = filters.threshold_otsu(rms_image)
    threshold = (1 + tolerance) * np.min(rms_image)
    mask = rms_image < threshold

    contours = measure.find_contours(rms_image, threshold)

    try:
        contour = sorted(contours, key=len)[-1]
    except IndexError:
        contour = np.array([0, 0])

    # [Averages shifts and errors from regions within the tolerated blocks]
    mean_shifts = [np.mean(shifted_image[mask, 0]), np.mean(shifted_image[mask, 1])]
    std_shifts = [np.std(shifted_image[mask, 0]), np.std(shifted_image[mask, 1])]
    mean_shift_norm = np.mean(shift_image_norm[mask])
    mean_error = np.mean(rms_image[mask])
    relative_shifts = np.abs(shift_image_norm - mean_shift_norm)

    # [calculates global shift, if it is better than the polled shift, or
    # if we do not have enough pollsters to fall back to then it does a global cross correlation!]
    mean_shifts_global, _, _ = phase_cross_correlation(
        img_1, img_2, upsample_factor=100
    )
    img_2_aligned_global = shift_image(img_2, shift)
    mean_error_global = np.sum(np.sum(np.abs(img_1 - img_2_aligned_global), axis=1))

    print_log(
        f"Block alignment error: {mean_error}, global alignment error: {mean_error_global}"
    )

    if (
        np.sum(mask) < min_number_pollsters
        or mean_error_global < mean_error
        or np.max(std_shifts) > shift_error_tolerance
    ):
        mean_shifts = mean_shifts_global
        mean_error = mean_error_global
        print_log("Falling back to global registration")

    print_log(
        f"*** Global XY shifts: {mean_shifts_global[0]:.2f} px | {mean_shifts_global[1]:.2f} px"
    )
    print_log(
        f"*** Mean polled XY shifts: {mean_shifts[0]:.2f}({std_shifts[0]:.2f}) px | {mean_shifts[1]:.2f}({std_shifts[1]:.2f}) px"
    )

    return np.array(mean_shifts), mean_error, relative_shifts, rms_image, contour


def compute_global_shift(
    image1_uncorrected, image2_uncorrected, lower_threshold, higher_threshold
):
    # [calculates unique translation for the entire image using cross-correlation]
    (
        shift,
        _,
        diffphase,
        lower_threshold,
        i_histogram,
        _,
        _,
        _,
    ) = align_2_images_cross_correlation(
        image1_uncorrected,
        image2_uncorrected,
        lower_threshold=lower_threshold,
        higher_threshold=higher_threshold,
    )

    return shift, diffphase, i_histogram, lower_threshold


def compute_shift_by_block(
    image1_uncorrected,
    image2_uncorrected,
    dict_block_size,
    tolerance,
):
    # [calculates block translations by cross-correlation and gets overall shift by polling]

    # normalizes images
    image1_uncorrected, image2_uncorrected = (
        np.float32(image1_uncorrected),
        np.float32(image2_uncorrected),
    )
    # matches histograms
    image2_uncorrected = np.float32(
        match_histograms(image2_uncorrected, image1_uncorrected)
    )
    # calculates block shifts and polls for most favourable shift
    block_size = (dict_block_size, dict_block_size)
    (
        shift,
        _,
        relative_shifts,
        rms_image,
        contour,
    ) = align_images_by_blocks(
        image1_uncorrected,
        image2_uncorrected,
        block_size,
        upsample_factor=100,
        min_number_pollsters=4,
        tolerance=tolerance,
    )
    diffphase = 0

    return (
        image1_uncorrected,
        image2_uncorrected,
        shift,
        diffphase,
        relative_shifts,
        rms_image,
        contour,
    )


def calcul_error(shifted_img, ref_img):
    shifted_img[shifted_img < 0] = 0
    error = np.sum(np.sum(np.abs(ref_img - shifted_img), axis=1))
    return error


def apply_registrations_to_filename(
    filename_to_process,
    dict_shifts,
    data_path,
    params: RegistrationParams,
    roi_name,
    projection_params: ProjectionParams,
):
    """
    Applies registration of filename_to_process

    Parameters
    ----------
    filename_to_process : string
        file to apply registration to
    dict_shifts : Dictionary
        contains the shifts to be applied to all rois

    Returns
    -------
    None.

    """
    # gets shift from dictionary

    label = os.path.basename(filename_to_process).split("_")[2]  # to FIX

    try:
        shift_array = dict_shifts[f"ROI:{roi_name}"][label]
    except KeyError:
        shift_array = None
        print_log(
            f"$ Could not find dictionary with alignment parameters for this ROI: ROI:{roi_name}, label: {label}"
        )

    if shift_array is not None:
        shift = np.asarray(shift_array)
        # loads 2D image and applies registration
        im_obj = Image()
        im_obj.load_image_2d(
            filename_to_process, data_path + os.sep + projection_params.folder
        )
        im_obj.data_2d = shift_image(im_obj.data_2d, shift)
        print_log(
            f"$ Image registered using ROI:{roi_name}, label:{label}, shift={shift}"
        )

        # saves registered 2D image
        im_obj.save_image_2d(
            data_path + os.sep + params.register_global_folder,
            tag="_2d_registered",
        )

    elif label == params.referenceFiducial:
        im_obj = Image()
        im_obj.load_image_2d(
            filename_to_process, data_path + os.sep + projection_params.folder
        )
        im_obj.save_image_2d(
            data_path + os.sep + params.register_global_folder,
            tag="_2d_registered",
        )
        print_log(f"$ Saving image for referenceRT ROI:{roi_name}, label:{label}")

    else:
        print_log(
            f"# No shift found in dictionary for ROI:{roi_name}, label:{label}",
            status="WARN",
        )


def apply_registrations_to_current_folder(
    data_path,
    current_param,
    params: RegistrationParams,
    roi_name,
    projection_params: ProjectionParams,
):
    """
    This function will
    - load masks, RNA and barcode 2D projected images,
    - apply registrations
    - save registered images as npy arrays

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    current_param : Parameters class

    Returns
    -------
    None.

    """
    files_folder = glob.glob(data_path + os.sep + "*.tif")
    print_log(f"> Processing Folder: {data_path}")
    dict_filename = (
        data_path
        + os.sep
        + params.register_global_folder
        + os.sep
        + "data"
        + os.sep
        + params.outputFile
        + ".json"
    )
    dict_shifts = load_json(dict_filename)
    if len(dict_shifts) == 0:
        print_log(f"# File with dictionary not found!: {dict_filename}")
        raise ValueError(f"# File with dictionary not found!: {dict_filename}")
    else:
        print_log(f"$ Dictionary File loaded: {dict_filename}")

    # generates lists of files to process
    current_param.find_files_to_process(files_folder)
    n_files = len(current_param.files_to_process)
    print_log(f"\n$ About to process {n_files} files\n")

    if len(current_param.files_to_process) > 0:
        # loops over files in file list
        for i, filename_to_process in enumerate(current_param.files_to_process):
            print_log(f"\n$ About to process file {i} \\ {n_files}")
            apply_registrations_to_filename(
                filename_to_process,
                dict_shifts,
                data_path,
                params,
                roi_name,
                projection_params,
            )


# =============================================================================
# IMAGE ALIGNMENT
# =============================================================================


def apply_xy_shift_3d_images(image, shift, parallel_execution=True):
    """
    Applies XY shift to a 3D stack

    Parameters
    ----------
    images : 3D numpy array
        image to process.

    Returns
    -------
    shifted 3D image.

    """
    client = try_get_client() if parallel_execution else None
    number_planes = image.shape[0]

    if client is None:
        print_log(f"> Shifting {number_planes} planes with 1 thread...")
        if len(image.shape) == 2:
            output = shift_image(image, shift)
        elif len(image.shape) == 3:
            shift_3d = np.zeros((3))
            shift_3d[0], shift_3d[1], shift_3d[2] = 0, shift[0], shift[1]
            output = shift_image(image, shift_3d)
        else:
            raise ValueError
    else:
        print_log(
            f"> Shifting {number_planes} planes using {len(client.scheduler_info()['workers'])} workers..."
        )

        image_list_scattered = scatter_3d_image(image)

        futures = [
            client.submit(shift_image, img, shift) for img in image_list_scattered
        ]

        output = reassemble_3d_image(client, futures, image.shape)

        del futures
        del image_list_scattered

    print_log("$ Done shifting 3D image.")

    return output


def image_block_alignment_3d(images, block_size_xy=256, upsample_factor=100):
    # sanity checks
    if len(images) < 2:
        sys.exit(f"# Error, number of images must be 2, not {len(images)}")

    # - break in blocks
    num_planes = images[0].shape[0]
    block_size = (num_planes, block_size_xy, block_size_xy)

    print_log("$ Breaking images into blocks")
    blocks = [view_as_blocks(x, block_shape=block_size).squeeze() for x in images]

    block_ref = blocks[0]
    block_target = blocks[1]

    # - loop thru blocks and calculates block shift in xyz:
    shift_matrices = [np.zeros(block_ref.shape[:2]) for _ in range(3)]

    for i in trange(block_ref.shape[0]):
        for j in range(block_ref.shape[1]):
            # - cross correlate in 3D to find 3D shift
            shifts_xyz, _, _ = phase_cross_correlation(
                block_ref[i, j], block_target[i, j], upsample_factor=upsample_factor
            )
            for matrix, _shift in zip(shift_matrices, shifts_xyz):
                matrix[i, j] = _shift

    return shift_matrices, block_ref, block_target


def combine_blocks_image_by_reprojection(
    block_ref, block_target, shift_matrices=None, axis1=0
):
    """
    This routine will overlap block_ref and block_target images block by block.
    block_ref will be used as a template.
    - block_target will be first translated in ZXY using the corresponding values in shift_matrices
    to realign each block
    - then an rgb image will be created with block_ref in the red channel, and the reinterpolated
    block_target block in the green channel.
    - the Blue channel is used for the grid to improve visualization of blocks.


    Parameters
    ----------
    block_ref : npy array
        return of view_as_blocks()
    block_target : npy array
        return of view_as_blocks()
    shift_matrices : list of npy arrays
        index 0 contains Z, index 1 X and index 2 Y
    axis1 : int
        axis used for the reprojection: The default is 0.
        - 0 means an XY projection
        - 1 an ZX projection
        - 2 an ZY projection

    Returns
    -------
    output : NPY array of size im_size x im_size x 3
        rgb image.
    ssim_as_blocks = NPY array of size number_blocks x number_blocks
        Structural similarity index between ref and target blocks
    """
    number_blocks = block_ref.shape[0]
    block_sizes = list(block_ref.shape[2:])
    block_sizes.pop(axis1)
    img_sizes = [x * number_blocks for x in block_sizes]

    # gets ranges for slicing
    slice_coordinates = [
        [range(x * block_size, (x + 1) * block_size) for x in range(number_blocks)]
        for block_size in block_sizes
    ]

    # creates output images
    output = np.zeros((img_sizes[0], img_sizes[1], 3))
    ssim_as_blocks = np.zeros((number_blocks, number_blocks))
    mse_as_blocks = np.zeros((number_blocks, number_blocks))
    nrmse_as_blocks = np.zeros((number_blocks, number_blocks))

    # blank image for blue channel to show borders between blocks
    blue = np.zeros(block_sizes)
    blue[0, :], blue[:, 0], blue[:, -1], blue[-1, :] = [0.5] * 4

    # reassembles image
    # takes one plane block
    for i, i_slice in enumerate(tqdm(slice_coordinates[0])):
        for j, j_slice in enumerate(slice_coordinates[1]):
            imgs = [block_ref[i, j]]
            if shift_matrices is not None:
                shift_3d = np.array(
                    [x[i, j] for x in shift_matrices]
                )  # gets 3D shift from block decomposition
                imgs.append(
                    shift_image(block_target[i, j], shift_3d)
                )  # realigns and appends to image list
            else:
                imgs.append(
                    block_target[i, j]
                )  # appends original target with no re-alignment

            imgs = [np.sum(x, axis=axis1) for x in imgs]  # projects along axis1
            imgs = [
                exposure.rescale_intensity(x, out_range=(0, 1)) for x in imgs
            ]  # rescales intensity values
            imgs = [
                image_adjust(x, lower_threshold=0.5, higher_threshold=0.9999)[0]
                for x in imgs
            ]  # adjusts pixel intensities

            nrmse_as_blocks[i, j] = normalized_root_mse(
                imgs[0], imgs[1], normalization="euclidean"
            )
            mse_as_blocks[i, j] = mean_squared_error(imgs[0], imgs[1])
            ssim_as_blocks[i, j] = ssim(
                imgs[0], imgs[1], data_range=imgs[1].max() - imgs[1].min()
            )

            imgs.append(blue)  # appends last channel with grid

            rgb = np.dstack(imgs)  # makes block rgb image

            output[i_slice[0] : i_slice[-1] + 1, j_slice[0] : j_slice[-1] + 1, :] = (
                rgb  # inserts block into final rgb stack
            )

    return output, ssim_as_blocks, mse_as_blocks, nrmse_as_blocks


def align_2_images_cross_correlation(
    image1_uncorrected,
    image2_uncorrected,
    lower_threshold=0.999,
    higher_threshold=0.9999999,
    upsample_factor=100,
):
    """
    Aligns 2 images by contrast adjust and cross correlation
    Parameters
    ----------
    img_reference : TYPE
        DESCRIPTION.
    img_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    shift : TYPE
        DESCRIPTION.
    error : TYPE
        DESCRIPTION.
    diffphase : TYPE
        DESCRIPTION.
    lower_threshold : TYPE
        DESCRIPTION.
    i_histogram : TYPE
        DESCRIPTION.
    image2_corrected : TYPE
        DESCRIPTION.
    image1_adjusted : TYPE
        DESCRIPTION.
    image2_adjusted : TYPE
        DESCRIPTION.
    image2_corrected_raw : TYPE
        DESCRIPTION.

    """

    (
        image1_adjusted,
        hist1_before,
        hist1_after,
        lower_cutoff1,
        _,
    ) = image_adjust(
        image1_uncorrected,
        lower_threshold=lower_threshold,
        higher_threshold=higher_threshold,
    )
    (
        image2_adjusted,
        hist2_before,
        hist2_after,
        lower_cutoff2,
        _,
    ) = image_adjust(
        image2_uncorrected,
        lower_threshold=lower_threshold,
        higher_threshold=higher_threshold,
    )

    # zips histograms
    lower_threshold = {"Im1": lower_cutoff1, "Im2": lower_cutoff2}
    i_histogram = {
        "Im1": (hist1_before, hist1_after),
        "Im2": (hist2_before, hist2_after),
    }

    # calculates shift
    shift, error, diffphase = phase_cross_correlation(
        image1_adjusted, image2_adjusted, upsample_factor=upsample_factor
    )

    # corrects image
    # The shift corresponds to the pixel offset relative to the reference image
    image2_corrected = shift_image(image2_adjusted, shift)
    image2_corrected = exposure.rescale_intensity(image2_corrected, out_range=(0, 1))

    return (
        shift,
        error,
        diffphase,
        lower_threshold,
        i_histogram,
        image2_corrected,
        image1_adjusted,
        image2_adjusted,
    )
