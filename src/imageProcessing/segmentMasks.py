#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:59:43 2020

@author: marcnol

File containing all functions responsible for segmentation of masks for Hi-M,
including DNA masks, barcodes, and fiducials

At the moment, fittings of the 2D positions of barcodes is also performed just
after image segmentation.

"""


# =============================================================================
# IMPORTS
# =============================================================================

# ---- stardist
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import time
import uuid

# ---- stardist
import matplotlib

# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Column, Table, vstack
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from csbdeep.data import PadAndCropResizer
from csbdeep.utils import normalize
from csbdeep.utils.tf import limit_gpu_memory
from dask.distributed import get_client
from matplotlib.path import Path
from photutils import (
    Background2D,
    DAOStarFinder,
    MedianBackground,
    deblend_sources,
    detect_sources,
    detect_threshold,
)
from photutils.segmentation.core import SegmentationImage
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi
from skimage import measure
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.util.apply_parallel import apply_parallel
from stardist import random_label_cmap
from stardist.models import StarDist2D, StarDist3D
from tqdm import trange

from core.dask_cluster import try_get_client
from core.parameters import SegmentationParams
from core.pyhim_logging import print_log, write_string_to_file
from core.saving import save_image_2d_cmd
from imageProcessing.imageProcessing import Image, reassemble_3d_image, scatter_3d_image

np.seterr(divide="ignore", invalid="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ignore tensorflow logging
matplotlib.rcParams["image.interpolation"] = "none"


# =============================================================================
# FUNCTIONS
# =============================================================================


def _show_image_sources(
    im, im1_bkg_substracted, x, y, flux, percent=99.5, vmin=0, vmax=2000
):
    fig, ax = plt.subplots()
    fig.set_size_inches((50, 50))

    norm = simple_norm(im, "sqrt", percent=percent)
    ax.imshow(im1_bkg_substracted, cmap="Greys", origin="lower", norm=norm)

    p_1 = ax.scatter(
        x,
        y,
        c=flux,
        s=50,
        facecolors="none",
        cmap="jet",
        marker="x",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(p_1, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlim(0, im.shape[1] - 1)
    ax.set_ylim(0, im.shape[0] - 1)

    return fig


def show_image_sources(
    im, im1_bkg_substracted, sources, markdown_filename, output_filename
):
    percent = 99.5
    flux = sources["flux"]
    x = sources["xcentroid"] + 0.5
    y = sources["ycentroid"] + 0.5

    fig = _show_image_sources(im, im1_bkg_substracted, x, y, flux, percent=percent)
    fig.savefig(f"{output_filename}_segmentedSources.png")
    plt.close(fig)

    write_string_to_file(
        markdown_filename,
        f"{os.path.basename(output_filename)}\n ![]({output_filename}_segmentedSources.png)\n",
        "a",
    )


def show_image_masks(im, segm_deblend, markdown_filename, output_filename):
    lbl_cmap = random_label_cmap()

    norm = ImageNormalize(stretch=SqrtStretch())
    cmap = lbl_cmap

    fig = plt.figure()
    fig.set_size_inches((30, 30))
    plt.imshow(im, cmap="Greys_r", origin="lower", norm=norm)
    plt.imshow(segm_deblend, origin="lower", cmap=cmap, alpha=0.5)
    plt.savefig(f"{output_filename}_segmentedMasks.png")
    plt.close()
    write_string_to_file(
        markdown_filename,
        f"{os.path.basename(output_filename)}\n ![]({output_filename}_segmentedMasks.png)\n",
        "a",
    )


def _segment_source_inhomog_background(
    im, threshold_over_std, fwhm, brightest, sigma_clip
):
    """
    Function that segments barcodes by estimating inhomogeneous background
    Parameters
    ----------
    im : NPY 2D
        image to be segmented

    Returns
    -------
    table : `astropy.table.Table` or `None`
    A table of found stars with the following parameters:

    * ``id``: unique object identification number.
    * ``xcentroid, ycentroid``: object centroid.
    * ``sharpness``: object sharpness.
    * ``roundness1``: object roundness based on symmetry.
    * ``roundness2``: object roundness based on marginal Gaussian
      fits.
    * ``npix``: the total number of pixels in the Gaussian kernel
      array.
    * ``sky``: the input ``sky`` parameter.
    * ``peak``: the peak, sky-subtracted, pixel value of the object.
    * ``flux``: the object flux calculated as the peak density in
      the convolved image divided by the detection threshold.  This
      derivation matches that of `DAOFIND`_ if ``sky`` is 0.0.
    * ``mag``: the object instrumental magnitude calculated as
      ``-2.5 * log10(flux)``.  The derivation matches that of
      `DAOFIND`_ if ``sky`` is 0.0.

    `None` is returned if no stars are found.

    img_bkc_substracted: 2D NPY array with background subtracted image
    """

    # estimates and removes inhomogeneous background
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        im,
        (64, 64),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    im1_bkg_substracted = im - bkg.background
    _, _, std = sigma_clipped_stats(im1_bkg_substracted, sigma=3.0)

    # estimates sources
    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_over_std * std,
        brightest=brightest,
        exclude_border=True,
    )
    sources = daofind(im1_bkg_substracted)

    return sources, im1_bkg_substracted


def segment_source_inhomog_background(im, seg_params: SegmentationParams):
    """
    Wrapper for function that segments barcodes by estimating inhomogeneous background
    Parameters
    ----------
    im : NPY 2D
        image to be segmented

    Returns
    -------
    table : `astropy.table.Table` or `None`
    A table of found stars with the following parameters:

    * ``id``: unique object identification number.
    * ``xcentroid, ycentroid``: object centroid.
    * ``sharpness``: object sharpness.
    * ``roundness1``: object roundness based on symmetry.
    * ``roundness2``: object roundness based on marginal Gaussian
      fits.
    * ``npix``: the total number of pixels in the Gaussian kernel
      array.
    * ``sky``: the input ``sky`` parameter.
    * ``peak``: the peak, sky-subtracted, pixel value of the object.
    * ``flux``: the object flux calculated as the peak density in
      the convolved image divided by the detection threshold.  This
      derivation matches that of `DAOFIND` if ``sky`` is 0.0.
    * ``mag``: the object instrumental magnitude calculated as
      ``-2.5 * log10(flux)``.  The derivation matches that of
      `DAOFIND` if ``sky`` is 0.0.

    `None` is returned if no stars are found.

    img_bkc_substracted: 2D NPY array with background subtracted image
    """

    threshold_over_std = seg_params.threshold_over_std
    fwhm = seg_params.fwhm
    brightest = seg_params.brightest  # keeps brightest sources

    # sigma_clip = SigmaClip(sigma=3.0)
    sigma_clip = SigmaClip(sigma=seg_params.background_sigma)

    sources, im1_bkg_substracted = _segment_source_inhomog_background(
        im, threshold_over_std, fwhm, brightest, sigma_clip
    )
    return sources, im1_bkg_substracted


def segment_source_flat_background(im, seg_params: SegmentationParams):
    """
    Segments barcodes using flat background
    Parameters
    ----------
    im : NPY 2D
        image to be segmented

    Returns
    -------
    table : `~astropy.table.Table` or `None`
    A table of found stars with the following parameters:

    * ``id``: unique object identification number.
    * ``xcentroid, ycentroid``: object centroid.
    * ``sharpness``: object sharpness.
    * ``roundness1``: object roundness based on symmetry.
    * ``roundness2``: object roundness based on marginal Gaussian
      fits.
    * ``npix``: the total number of pixels in the Gaussian kernel
      array.
    * ``sky``: the input ``sky`` parameter.
    * ``peak``: the peak, sky-subtracted, pixel value of the object.
    * ``flux``: the object flux calculated as the peak density in
      the convolved image divided by the detection threshold.  This
      derivation matches that of `DAOFIND` if ``sky`` is 0.0.
    * ``mag``: the object instrumental magnitude calculated as
      ``-2.5 * log10(flux)``.  The derivation matches that of
      `DAOFIND` if ``sky`` is 0.0.

    `None` is returned if no stars are found.

    img_bkc_substracted: 2D NPY array with background subtracted image
    """

    threshold_over_std = seg_params.threshold_over_std
    fwhm = seg_params.fwhm

    # removes background
    _, median, std = sigma_clipped_stats(im, seg_params.background_sigma)
    im1_bkg_substracted = im - median

    # estimates sources
    daofind = DAOStarFinder(
        fwhm=fwhm, threshold=float(threshold_over_std * std), exclude_border=True
    )
    sources = daofind(im - median)

    return sources, im1_bkg_substracted


def tessellate_masks(segm_deblend):
    """
    * takes a labeled mask (background 0, nuclei labeled 1, 2, ...)
    * calls get_tessellation(xy, img_shape)
    * returns the tessellated mask and the voronoi data structure

    Parameters
    ----------
    segm_deblend : TYPE
        Labeled mask.

    Returns
    -------
    voronoi_data : TYPE
        DESCRIPTION.
    mask_voronoi : TYPE
        DESCRIPTION.

    """
    start_time = time.time()

    # get centroids
    mask_labeled = segm_deblend.data
    mask_binary = mask_labeled.copy()
    mask_binary[mask_binary > 0] = 1

    regions = regionprops(mask_labeled)

    num_masks = np.max(mask_labeled)
    centroid = np.zeros((num_masks + 1, 2))  # +1 as labels run from 0 to max

    for props in regions:
        y_0, x_0 = props.centroid
        label = props.label
        centroid[label, :] = x_0, y_0

    # tessellation
    # remove first centroid (this is the background label)
    xy = centroid[1:, :]
    voronoi_data = get_tessellation(xy, mask_labeled.shape)

    # add some clipping to the tessellation
    # gaussian blur and thresholding; magic numbers!
    mask_blurred = gaussian_filter(mask_binary.astype("float64"), sigma=20)
    mask_blurred = mask_blurred > 0.01

    # convert tessellation to mask
    np.random.seed(42)

    mask_voronoi = np.zeros(mask_labeled.shape, dtype="int64")

    nx, ny = mask_labeled.shape

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    # currently takes 1min for approx 600 polygons
    for label in range(num_masks):  # label is shifted by -1 now
        mask_id = label + 1

        idx_vor_region = voronoi_data.point_region[label]
        idx_vor_vertices = voronoi_data.regions[
            idx_vor_region
        ]  # list of indices of the Voronoi vertices

        vertices = np.full((len(idx_vor_vertices), 2), np.NaN)
        drop_vert = False
        # pylint: disable-next=consider-using-enumerate
        for i in range(len(idx_vor_vertices)):
            idx = idx_vor_vertices[i]
            if (
                idx == -1
            ):  # this means a "virtual point" at infinity as the vertex is not closed
                drop_vert = True
                print_log('$ Detected "virtual point" at infinity. Skipping this mask.')
                break
            vertices[i, :] = voronoi_data.vertices[idx]

        if drop_vert:  # region is not bounded
            continue

        poly_path = Path(vertices)
        mask = poly_path.contains_points(points)
        mask = mask.reshape((ny, nx))
        mask_voronoi[mask & mask_blurred] = mask_id

    # print_log("--- Took {:.2f}s seconds ---".format(time.time() - start_time))
    print_log(f"$ Tessellation took {(time.time() - start_time):.2f}s seconds.")

    return voronoi_data, mask_voronoi


def get_tessellation(xy, img_shape):
    """
    * runs the actual tessellation based on the xy position of the markers in an image of given shape

    # follow this tutorial
    # https://hpaulkeeler.com/voronoi-dirichlet-tessellations/
    # https://github.com/hpaulkeeler/posts/blob/master/PoissonVoronoi/PoissonVoronoi.py

    # changes:
    # added dummy points outside of the image corners (in quite some distance)
    # they are supposed "catch" all the vertices that end up at infinity
    # follows an answer given here
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    # Attributes
    #    points ndarray of double, shape (npoints, ndim)
    #        Coordinates of input points.
    #
    #    vertices ndarray of double, shape (nvertices, ndim)
    #        Coordinates of the Voronoi vertices.
    #
    #    ridge_points ndarray of ints, shape (nridges, 2)
    #        Indices of the points between which each Voronoi ridge lies.
    #
    #    ridge_vertices list of list of ints, shape (nridges, *)
    #        Indices of the Voronoi vertices forming each Voronoi ridge.
    #
    #    regions list of list of ints, shape (nregions, *)
    #        Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
    #
    #    point_region list of ints, shape (npoints)
    #        Index of the Voronoi region for each input point. If qhull option “Qc” was not specified, the list will contain -1 for points that are not associated with a Voronoi region.
    #
    #    furthest_site
    #        True if this was a furthest site triangulation and False if not.
    #        New in version 1.4.0.

    Parameters
    ----------
    xy : TYPE
        DESCRIPTION.
    img_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    voronoi_data : TYPE
        DESCRIPTION.


    """

    x_center, y_center = np.array(img_shape) / 2
    x_max, y_max = np.array(img_shape)

    corner1 = [x_center - 100 * x_max, y_center - 100 * y_max]
    corner2 = [x_center + 100 * x_max, y_center - 100 * y_max]
    corner3 = [x_center - 100 * x_max, y_center + 100 * y_max]
    corner4 = [x_center + 100 * x_max, y_center + 100 * y_max]

    xy = np.append(xy, [corner1, corner2, corner3, corner4], axis=0)

    # perform Voroin tesseslation
    return Voronoi(xy)


def segment_mask_inhomog_background(im, seg_params: SegmentationParams):
    """
    Function used for segmenting masks with the ASTROPY library that uses image processing

    Parameters
    ----------
    im : 2D np array
        image to be segmented.

    Returns
    -------
    segm_deblend: 2D np array where each pixel contains the label of the mask segmented. Background: 0

    """
    # removes background
    threshold = detect_threshold(im, nsigma=2.0)
    sigma_clip = SigmaClip(sigma=seg_params.background_sigma)

    bkg_estimator = MedianBackground()
    bkg = Background2D(
        im,
        (64, 64),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    threshold = bkg.background + (
        seg_params.threshold_over_std * bkg.background_rms
    )  # background-only error image, typically 1.0

    sigma = seg_params.fwhm * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    data = convolve(im, kernel, mask=None, normalize_kernel=True)
    # estimates masks and deblends
    segm = detect_sources(data, threshold, npixels=seg_params.area_min)

    # removes masks too close to border
    segm.remove_border_labels(
        border_width=10
    )  # TODO: parameter to add to parameters.json ?

    segm_deblend = deblend_sources(
        data,
        segm,
        npixels=seg_params.area_min,  # typically 50 for masks
        nlevels=32,
        contrast=0.001,  # try 0.2 or 0.3
        relabel=True,
    )

    # removes Masks too big or too small
    for label in segm_deblend.labels:
        # take regions with large enough areas
        area = segm_deblend.get_area(label)
        # print_log('label {}, with area {}'.format(label,area))
        if area < seg_params.area_min or area > seg_params.area_max:
            segm_deblend.remove_label(label=label)
            # print_log('label {} removed'.format(label))

    # relabel so masks numbers are consecutive
    segm_deblend.relabel_consecutive()

    return segm_deblend


def segment_mask_stardist(im, seg_params: SegmentationParams):
    """
    Function used for segmenting masks with the STARDIST package that uses Deep Convolutional Networks

    Parameters
    ----------
    im : 2D np array
        image to be segmented.

    Returns
    -------
    segm_deblend: 2D np array where each pixel contains the label of the mask segmented. Background: 0

    """

    np.random.seed(6)
    sigma = seg_params.fwhm * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()

    n_channel = 1 if im.ndim == 2 else im.shape[-1]
    axis_norm = (0, 1)  # normalize channels independently

    if n_channel > 1:
        print_log(
            f'> Normalizing image channels {"jointly" if axis_norm is None or 2 in axis_norm else "independently"}.'
        )
    if seg_params.stardist_basename is not None and os.path.exists(
        seg_params.stardist_basename
    ):
        base_dir = seg_params.stardist_basename
    else:
        base_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, "stardist_models"
        )
    if seg_params.stardist_network is not None and os.path.exists(
        os.path.join(base_dir, seg_params.stardist_network)
    ):
        model_name = seg_params.stardist_network
    else:
        model_name = "DAPI_2D_stardist_nc14_nrays64_epochs40_grid2"
    model = StarDist2D(None, name=model_name, basedir=base_dir)

    img = normalize(im, 1, 99.8, axis=axis_norm)
    labeled, _ = model.predict_instances(img)

    # estimates masks and deblends
    segm = SegmentationImage(labeled)

    # removes masks too close to border
    segm.remove_border_labels(border_width=10)
    segm_deblend = segm

    # removes Masks too big or too small
    for label in segm_deblend.labels:
        # take regions with large enough areas
        area = segm_deblend.get_area(label)
        if area < seg_params.area_min or area > seg_params.area_max:
            segm_deblend.remove_label(label=label)

    # relabel so masks numbers are consecutive
    segm_deblend.relabel_consecutive()

    return segm_deblend, labeled


def make_segmentations(
    file_name,
    current_param,
    data_path,
    seg_params: SegmentationParams,
    align_folder,
    label,
):
    root_filename = os.path.basename(file_name).split(".")[0]
    filename_2d_aligned = align_folder + os.sep + root_filename + "_2d_registered.npy"

    print_log(f"> searching for {filename_2d_aligned}")
    if os.path.exists(filename_2d_aligned):  # file exists
        # TODO: use decode_file_parts with the regex --> should be done with DataManager
        roi = os.path.basename(file_name).split("_")[3]

        # loading registered 2D projection
        im_obj = Image()
        im_obj.load_image_2d(
            file_name,
            align_folder[:-5],  # remove extra "/data", it's temporary for refactoring
            tag="_2d_registered",
        )
        im = im_obj.data_2d
        print_log(
            f"> [{label}] Loaded 2D registered file: {os.path.basename(file_name)}"
        )

        ##########################################
        #               Segments barcodes
        ##########################################

        if label == "barcode" and [i for i in root_filename.split("_") if "RT" in i]:
            output_filename = (
                data_path
                + os.sep
                + seg_params.localize_2d_folder
                + os.sep
                + root_filename
            )
            segmentation_method = seg_params.background_method
            print_log(f"\n$ Segmenting barcodes using method: {segmentation_method }")
            if segmentation_method == "flat":
                output = segment_source_flat_background(im, seg_params)
            elif segmentation_method == "inhomogeneous":
                output, im1_bkg_substracted = segment_source_inhomog_background(
                    im, seg_params
                )
                # show results
                show_image_sources(
                    im,
                    im1_bkg_substracted,
                    output,
                    current_param.param_dict["fileNameMD"],
                    output_filename,
                )
            else:
                print_log(
                    f"# Method <{segmentation_method}> not available for barcode segmentation!"
                )
                return Table()
            # [ formats results Table for output by adding buid, barcode_id, CellID and roi]

            # buid
            buid = [str(uuid.uuid4()) for _ in range(len(output))]
            col_buid = Column(buid, name="Buid", dtype=str)

            # barcode_id, cellID and roi
            barcode_id = os.path.basename(file_name).split("_")[2].split("RT")[1]
            col_roi = Column(int(roi) * np.ones(len(output)), name="ROI #", dtype=int)
            col_barcode = Column(
                int(barcode_id) * np.ones(len(output)), name="Barcode #", dtype=int
            )
            col_cell_id = Column(np.zeros(len(output)), name="CellID #", dtype=int)
            zcoord = Column(
                np.nan * np.zeros(len(output)), name="zcentroid", dtype=float
            )

            if output[0] is not None:
                # adds to table
                output.add_column(col_barcode, index=0)
                output.add_column(col_roi, index=0)
                output.add_column(col_buid, index=0)
                output.add_column(col_cell_id, index=2)
                output.add_column(zcoord, index=5)

            # changes format of table
            # for col in output.colnames:
            #    output[col].info.format = '%.8g'  # for consistent table output

        elif label in ("DAPI", "mask"):
            output_filename = (
                data_path + os.sep + seg_params.mask_2d_folder + os.sep + root_filename
            )
            if seg_params.background_method == "flat":
                output = segment_mask_inhomog_background(im, seg_params)
            elif seg_params.background_method == "inhomogeneous":
                output = segment_mask_inhomog_background(im, seg_params)
            elif seg_params.background_method == "stardist":
                output, labeled = segment_mask_stardist(im, seg_params)
            else:
                print_log(
                    "# segmentedObjects/background_method not specified in json file"
                )
                output = np.zeros(1)
                return output

            if seg_params.tessellation:
                _, output = tessellate_masks(output)

            # show results
            if "labeled" in locals():
                output_filename_stardist = (
                    data_path
                    + os.sep
                    + seg_params.mask_2d_folder
                    + os.sep
                    + root_filename
                    + "_stardist"
                )
                show_image_masks(
                    im,
                    labeled,
                    current_param.param_dict["fileNameMD"],
                    output_filename_stardist,
                )
                # saves output 2d zProjection as matrix
                save_image_2d_cmd(labeled, f"{output_filename}_Masks")
            else:
                show_image_masks(
                    im,
                    output,
                    current_param.param_dict["fileNameMD"],
                    output_filename,
                )
                # saves output 2d zProjection as matrix
                save_image_2d_cmd(output, f"{output_filename}_Masks")

        else:
            output = []
        del im_obj

        return output
    else:
        print_log(f"# 2D aligned file does not exist:{filename_2d_aligned}")
        print_log(f"\t{os.path.exists(filename_2d_aligned)}")
        return []


def segment_masks(
    current_param,
    data_path,
    params: SegmentationParams,
    align_folder,
    label,
    file_name=None,
):
    session_name = "segmentMasks"

    # processes folders and files
    print_log(f"\n===================={session_name}:{label}====================\n")

    write_string_to_file(
        current_param.param_dict["fileNameMD"],
        f"""## {session_name}: {label}\n""",
        "a",
    )
    barcodes_coordinates = Table()

    files_folder = glob.glob(data_path + os.sep + "*.tif")

    # generates lists of files to process
    current_param.find_files_to_process(files_folder)
    print_log(f"> Processing Folder: {data_path}")
    print_log(f"> Files to Segment: {len(current_param.files_to_process)}\n")

    barcode_out_file = (
        data_path
        + os.sep
        + params.localize_2d_folder
        + os.sep
        + "data"
        + os.sep
        + params.outputFile
        + "_"
        + label
        + ".dat"
    )
    if current_param.param_dict["parallel"]:
        # running in parallel mode
        client = get_client()
        futures = []

        for filename_to_process in current_param.files_to_process:
            if (
                file_name is None
                or (
                    file_name is not None
                    and os.path.basename(file_name)
                    == os.path.basename(filename_to_process)
                )
            ) and label != "fiducial":
                futures.append(
                    client.submit(
                        make_segmentations,
                        filename_to_process,
                        current_param,
                        data_path,
                        params,
                        align_folder,
                        label,
                    )
                )

        print_log(f"Waiting for {len(futures)} results to arrive")

        results = client.gather(futures)

        if label == "barcode":
            # gathers results from different barcodes and rois
            print_log(f"Retrieving {len(results)} results from cluster")
            detected_spots = []
            for result in results:
                detected_spots.append(len(result))
                barcodes_coordinates = vstack([barcodes_coordinates, result])

                # saves results together into a single Table
                barcodes_coordinates.write(
                    barcode_out_file, format="ascii.ecsv", overwrite=True
                )
            print_log(f"$ File {barcode_out_file} written to file.")
            print_log(f'$ Detected spots: {",".join([str(x) for x in detected_spots])}')

    else:
        for filename_to_process in current_param.files_to_process:
            if (
                file_name is None
                or (
                    file_name is not None
                    and os.path.basename(file_name)
                    == os.path.basename(filename_to_process)
                )
            ) and label != "fiducial":
                # running in sequential mode
                output = make_segmentations(
                    filename_to_process,
                    current_param,
                    data_path,
                    params,
                    align_folder,
                    label,
                )

                # gathers results from different barcodes and rois
                if label == "barcode":
                    barcodes_coordinates = vstack([barcodes_coordinates, output])
                    barcodes_coordinates.write(
                        barcode_out_file, format="ascii.ecsv", overwrite=True
                    )
                    print_log(f"$ File {barcode_out_file} written to file.")

    return 0


# =============================================================================
# SEGMENTATION FUNCTIONS
# =============================================================================


def _segment_2d_image_by_thresholding(
    image_2d,
    threshold_over_std=10,
    area_min=3,
    area_max=1000,
    nlevels=64,
    contrast=0.001,
    kernel=None,
):
    # makes threshold matrix
    threshold = np.zeros(image_2d.shape)
    threshold[:] = threshold_over_std * image_2d.max() / 100
    if kernel is not None:
        data = convolve(image_2d, kernel, mask=None, normalize_kernel=True)
    # segments objects
    segm = detect_sources(
        data,
        threshold,
        npixels=area_min,
    )

    if segm.nlabels <= 0:
        # returns empty image as no objects were detected
        return segm.data
    # removes masks too close to border
    segm.remove_border_labels(border_width=10)
    if segm.nlabels <= 0:
        # returns empty image as no objects were detected
        return segm.data

    segm_deblend = deblend_sources(
        data,
        segm,
        npixels=area_min,  # watch out, this is per plane!
        nlevels=nlevels,
        contrast=contrast,
        relabel=True,
        mode="exponential",
    )
    if segm_deblend.nlabels > 0:
        # removes Masks too big or too small
        for label in segm_deblend.labels:
            # take regions with large enough areas
            area = segm_deblend.get_area(label)
            if area < area_min or area > area_max:
                segm_deblend.remove_label(label=label)

        # relabel so masks numbers are consecutive
        # segm_deblend.relabel_consecutive()

    # image_2d_segmented = segm.data % changed during recoding function
    image_2d_segmented = segm_deblend.data

    image_2d_segmented[image_2d_segmented > 0] = 1
    return image_2d_segmented


def _segment_3d_volumes_stardist(
    image_3d,
    deblend_3d=False,
    axis_norm=(0, 1, 2),
    model_dir="/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/Annotated_data/data_loci_small/models/",
    model_name="stardist_18032021_single_loci",
):
    number_planes = image_3d.shape[0]

    print_log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print_log(f"> Segmenting {number_planes} planes using 1 worker...")
    print_log(f"> Loading model {model_name} from {model_dir}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = StarDist3D(None, name=model_name, basedir=model_dir)
    limit_gpu_memory(None, allow_growth=True)

    im = normalize(image_3d, 1, 99.8, axis=axis_norm)
    l_x = im.shape[1]

    if l_x < 1000:
        labels, _ = model.predict_instances(im)

    else:
        resizer = PadAndCropResizer()
        axes = "ZYX"

        im = resizer.before(im, axes, model._axes_div_by(axes))
        labels, _ = model.predict_instances(im, n_tiles=(1, 8, 8))
        labels = resizer.after(labels, axes)

    mask = np.array(labels > 0, dtype=int)

    # Now we want to separate objects in 3D using watersheding
    labeled_image = _deblend_3d_segmentation(mask) if deblend_3d else labels
    return mask, labeled_image


def _segment_3d_volumes_by_thresholding(
    image_3d,
    threshold_over_std=10,
    sigma=3,
    box_size=(32, 32),
    area_min=3,
    area_max=1000,
    nlevels=64,
    contrast=0.001,
    deblend_3d=False,
    parallel_execution=True,
):
    client = try_get_client() if parallel_execution else None
    number_planes = image_3d.shape[0]

    kernel = Gaussian2DKernel(sigma, x_size=sigma, y_size=sigma)
    kernel.normalize()

    print_log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    parallel = True
    if client is None:
        parallel = False
    elif len(client.scheduler_info()["workers"]) < 1:
        parallel = False
        print_log("# Failed getting workers. Report of scheduler:")
        for key in client.scheduler_info().keys():
            print_log(f"{key}:{client.scheduler_info()[key]}")

    if not parallel:
        print_log(f"> Segmenting {number_planes} planes using 1 worker...")

        output = np.zeros(image_3d.shape)

        for z in trange(number_planes):
            image_2d = image_3d[z, :, :]
            image_2d_segmented = _segment_2d_image_by_thresholding(
                image_2d,
                threshold_over_std=threshold_over_std,
                area_min=area_min,
                area_max=area_max,
                nlevels=nlevels,
                contrast=contrast,
                kernel=kernel,
            )
            output[z, :, :] = image_2d_segmented

    else:
        print_log(
            f"> Segmenting {number_planes} planes using {len(client.scheduler_info()['workers'])} workers..."
        )

        image_list_scattered = scatter_3d_image(image_3d)

        futures = [
            client.submit(
                _segment_2d_image_by_thresholding,
                img,
                threshold_over_std=threshold_over_std,
                sigma=sigma,
                box_size=box_size,
                area_min=area_min,
                area_max=area_max,
                nlevels=nlevels,
                contrast=contrast,
                deblend_3d=deblend_3d,
                kernel=kernel,
            )
            for img in image_list_scattered
        ]

        output = reassemble_3d_image(client, futures, image_3d.shape)

        del futures, image_list_scattered

    labels = measure.label(output)

    # Now we want to separate objects in 3D using watersheding
    if deblend_3d:
        labels = _deblend_3d_segmentation(output)

    return output, labels


def _deblend_3d_segmentation(binary):
    """
    Deblends objects in 3D using watersheding

    Parameters
    ----------
    binary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    binary = binary > 0
    print_log(" > Constructing distance matrix from 3D binary mask...")

    distance = apply_parallel(ndi.distance_transform_edt, binary)

    print_log(" > Deblending sources in 3D by watersheding...")
    coords = peak_local_max(distance, footprint=np.ones((10, 10, 25)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    labels = watershed(-distance, markers, mask=binary)
    return labels


def _segment_3d_masks(
    image_3d,
    axis_norm=(0, 1, 2),
    pmin=1,
    pmax=99.8,
    model_dir=None,
    model_name=None,
):
    """
    Parameters
    ----------
    image_3d : numpy ndarray (N-dimensional array)
        3D raw image to be segmented

    model_dir : List of strings, optional
        paths of all models directory, the default is None

    model_name : List of strings, optional
        names of all models, the default is None

    """

    np.random.seed(6)

    number_planes = image_3d.shape[0]

    print_log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print_log(f"> Segmenting {number_planes} planes using 1 worker...")
    print_log(f"> Loading model {model_name} from {model_dir}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # why do we need this?

    # Load the model
    # --------------

    model = StarDist3D(None, name=model_name, basedir=model_dir)
    limit_gpu_memory(None, allow_growth=True)

    im = normalize(image_3d, pmin=pmin, pmax=pmax, axis=axis_norm)
    l_x = im.shape[1]

    if l_x < 351:  # what is this value? should it be a k-arg?
        labels, _ = model.predict_instances(im)

    else:
        resizer = PadAndCropResizer()
        axes = "ZYX"

        im = resizer.before(im, axes, model._axes_div_by(axes))
        labels, _ = model.predict_instances(im, n_tiles=(1, 8, 8))
        labels = resizer.after(labels, axes)

    mask = np.array(labels > 0, dtype=int)
    mask[mask > 0] = 1

    return mask, labels
