#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for common image processing
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import SigmaClip
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import Background2D, MedianBackground
from skimage import exposure, io
from tqdm import trange

from core.dask_cluster import try_get_client
from core.pyhim_logging import print_log
from core.saving import save_image_2d_cmd

np.seterr(divide="ignore", invalid="ignore")

# =============================================================================
# CLASSES
# =============================================================================


class Image:
    def __init__(self, current_log=None):
        self.log = current_log
        self.data = []
        self.file_name = ""
        self.data_2d = np.zeros((1, 1))
        self.stage_coordinates = [0.0, 0.0]
        self.image_size = -1
        self.focus_plane = -1
        self.extension = ""
        self.z_range = None
        self.focal_plane_matrix = None

    # read an image as a numpy array
    def load_image(self, file_name):
        self.data = io.imread(file_name).squeeze()
        self.file_name = file_name
        self.image_size = self.data.shape
        self.extension = file_name.split(".")[-1]

    # save 2D projection as numpy array
    def save_image_2d(self, root_folder, tag="_2d"):
        file_name = self.get_image_filename(root_folder, tag)
        save_image_2d_cmd(self.data_2d, file_name)

    def get_image_filename(self, root_folder, tag):
        file_name = (
            root_folder + os.sep + os.path.basename(self.file_name).split(".")[0] + tag
        )
        return file_name

    # read an image as a numpy array
    def load_image_2d(self, file_name, master_folder, tag="_2d"):
        self.file_name = file_name
        folder = master_folder + os.sep + "data"
        file_name = f"{self.get_image_filename(folder, tag)}.npy"

        self.data_2d = np.load(file_name)
        print_log(f"$ Loading from disk:{os.path.basename(file_name)}")

    # displays image and shows it
    def show_image(
        self,
        show=False,
        size=(10, 10),
        output_name="tmp.png",
        save=True,
        normalization="stretch",
    ):
        fig = plt.figure()
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()

        if normalization == "simple":
            norm = simple_norm(self.data_2d, "sqrt", percent=99.9)
        else:
            norm = ImageNormalize(stretch=SqrtStretch())

        ax.set_title("2D Data")

        if show:
            fig.add_axes(ax)
            ax.imshow(self.data_2d, origin="lower", cmap="Greys_r", norm=norm)
            return ax

        if save:
            fig.add_axes(ax)
            ax.imshow(self.data_2d, origin="lower", cmap="Greys_r", norm=norm)
            fig.savefig(output_name)
            plt.close(fig)


def scatter_3d_image(image):
    """
    splits 3D image plane by plane and scatteres them to a cluster

    Parameters
    ----------
    image : numpy array
        3D image.

    Returns
    -------
    image_list_scattered :  List, dict, iterator, or queue of futures matching the type of input.
        scattered image.

    """
    number_planes = image.shape[0]
    return [image[z, :, :] for z in range(number_planes)]


def reassemble_3d_image(client, futures, output_shape):
    """
    waits for futures to arrive
    collects them into a results list
    reassembles 3D image plane by plane

    Parameters
    ----------
    client : dask CLient()
        result of get_client()
    futures : []
        list of futures
    output_shape : tuple
        result of image.shape

    Returns
    -------
    output : numpy array
        contains reassembled 3D image.

    """
    results = client.gather(futures)
    print_log(f" > Retrieving {len(results)} results from cluster")

    output = np.zeros(output_shape)
    for z, result in enumerate(results):
        output[z, :, :] = result

    del results
    return output


# =============================================================================
# CONTRAST and PIXEL INTENSITY NORMALIZATION, INHOMOGENEOUS BACKGROUND
# =============================================================================


def preprocess_3d_image(x, lower_threshold, higher_threshold, parallel_execution=True):
    """
    3D stack pre-processing:
        - rescales intensities to 0->1
        - removes inhomogeneous background plane by plane
        - adjusts image levels by thresholding

    Parameters
    ----------
    x : numpy array
        3D image.
    lower_threshold : float
        lower threshold for adjusting image levels.
    higher_threshold : float
        higher threshold for adjusting image levels..

    Returns
    -------
    image : numpy array
        pre-processed 3D image.

    """
    image = exposure.rescale_intensity(x, out_range=(0, 1))

    # print_log("Removing inhomogeneous background...")
    image = _remove_inhomogeneous_background(
        image, parallel_execution=parallel_execution
    )

    # print_log("Rescaling grey levels...")
    image = image_adjust(
        image, lower_threshold=lower_threshold, higher_threshold=higher_threshold
    )[0]

    return image


def image_adjust(image, lower_threshold=0.3, higher_threshold=0.9999):
    """
    Adjust intensity levels:
        - rescales exposures
        - gets histogram of pixel intensities to define cutoffs
        - applies thresholds

    Parameters
    ----------
    image : numpy array
        input 3D image.
    lower_threshold : float, optional
        lower threshold for adjusting image levels. The default is 0.3.
    higher_threshold : float, optional
        higher threshold for adjusting image levels.. The default is 0.9999.

    Returns
    -------
    image1 : numpy array
        adjusted 3D image.
    hist1_before : numpy array
        histogram of pixel intensities before adjusting levels.
    hist1 : numpy array
        histogram of pixel intensities after adjusting levels.
    lower_cutoff : float
        lower cutoff used for thresholding.
    higher_cutoff : float
        higher cutoff used for thresholding.

    """
    # print_log("> Rescaling grey levels...")

    # rescales image to [0,1]
    image1 = exposure.rescale_intensity(image, out_range=(0, 1))

    # calculates histogram of intensities
    hist1_before = exposure.histogram(image1)

    hist_sum = np.zeros(len(hist1_before[0]))
    for i in range(len(hist1_before[0]) - 1):
        hist_sum[i + 1] = hist_sum[i] + hist1_before[0][i]

    sum_normalized = hist_sum / hist_sum.max()
    lower_cutoff = np.where(sum_normalized > lower_threshold)[0][0] / 255
    higher_cutoff = np.where(sum_normalized > higher_threshold)[0][0] / 255

    # adjusts image intensities from (lower_threshold,higher_threshold) --> [0,1]
    image1 = exposure.rescale_intensity(
        image1, in_range=(lower_cutoff, higher_cutoff), out_range=(0, 1)
    )

    # calculates histogram of intensities of adjusted image
    hist1 = exposure.histogram(image1)

    return image1, hist1_before, hist1, lower_cutoff, higher_cutoff


def _remove_inhomogeneous_background(
    im,
    box_size=(32, 32),
    filter_size=(3, 3),
    parallel_execution=True,
    background=False,
):
    """
    wrapper to remove inhomogeneous backgrounds for 2D and 3D images

    Parameters
    ----------
    im : numpy array
        input image.
    box_size : tuple of ints, optional
        size of box_size used for block decomposition. The default is (32, 32).
    filter_size : tuple of ints, optional
        Size of gaussian filter used for smoothing results. The default is (3, 3).

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    if len(im.shape) == 2:
        output = _remove_inhomogeneous_background_2d(
            im,
            filter_size=filter_size,
            background=background,
        )
    elif len(im.shape) == 3:
        output = _remove_inhomogeneous_background_3d(
            im,
            box_size=box_size,
            filter_size=filter_size,
            parallel_execution=parallel_execution,
            background=background,
        )
    else:
        return None
    return output


def _remove_inhomogeneous_background_2d(im, filter_size=(3, 3), background=False):
    """
    Calls Background2D() from ASTROPY to perform background subtraction in a 2D image

    Parameters
    ----------
    im : numpy array
        2D input image.
    filter_size : tuple of ints, optional
        Size of gaussian filter used for smoothing results. The default is (3, 3).
    background : boolean, optional
        if True returns the subtracted image and the background
        otherwise only the former. The default is False.

    Returns
    -------
    numpy array
        background subtracted 2D image.

    """
    print_log("Removing inhomogeneous background from 2D image...")

    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        im,
        (64, 64),
        filter_size=filter_size,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    im1_bkg_substracted = im - bkg.background

    return (im1_bkg_substracted, bkg) if background else im1_bkg_substracted


def _remove_inhomogeneous_background_3d(
    image_3d,
    box_size=(64, 64),
    filter_size=(3, 3),
    parallel_execution=True,
    background=False,
):
    """
    Wrapper to remove inhomogeneous background in a 3D image by recursively calling _remove_inhomogeneous_background_2d():
        - addresses output
        - iterates over planes and calls _remove_inhomogeneous_background_2d in each plane
        - reassembles results into a 3D image

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    box_size : tuple of ints, optional
        size of box_size used for block decomposition. The default is (32, 32).
    filter_size : tuple of ints, optional
        Size of gaussian filter used for smoothing results. The default is (3, 3).

    Returns
    -------
    output : numpy array
        processed 3D image.

    """
    client = try_get_client() if parallel_execution else None
    number_planes = image_3d.shape[0]
    output = np.zeros(image_3d.shape)

    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    if client is not None:
        print_log(
            f"> Removing inhomogeneous background from {number_planes} planes using {len(client.scheduler_info()['workers'])} workers..."
        )
        image_list = [image_3d[z, :, :] for z in range(number_planes)]
        # image_list_scattered = client.scatter(image_list)

        futures = [
            client.submit(
                Background2D,
                img,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            for img in image_list
        ]

        results = client.gather(futures)
        print_log(f" > Retrieving {len(results)} results from cluster")

        for z, img, bkg in zip(range(number_planes), image_list, results):
            output[z, :, :] = img - bkg.background
        del results, futures, image_list
        # del image_list_scattered

    else:
        print_log(
            f"> Removing inhomogeneous background from {number_planes} planes using 1 worker..."
        )
        z_range = trange(number_planes)
        for z in z_range:
            image_2d = image_3d[z, :, :]
            bkg = Background2D(
                image_2d,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            output[z, :, :] = image_2d - bkg.background

    return (output, bkg.background) if background else output
