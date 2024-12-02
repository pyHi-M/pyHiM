#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for common image processing
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from matplotlib import ticker
from skimage import color
from skimage.util.shape import view_as_blocks
from stardist import random_label_cmap
from tifffile import imsave
from tqdm import trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ignore tensorflow logging
np.seterr(divide="ignore", invalid="ignore")


def plot_raw_images_and_labels(image, label):
    """
    Parameters
    ----------
    image : List of numpy ndarray (N-dimensional array)
        3D raw image of format .tif

    label : List of numpy ndarray (N-dimensional array)
        3D labeled image of format .tif
    """

    cmap = random_label_cmap()

    moy = np.mean(image, axis=0)
    lbl_moy = np.max(label, axis=0)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((50, 50))
    ax = axes.ravel()
    titles = ["raw image", "projected labeled image"]

    ax[0].imshow(moy, cmap="Greys_r", origin="lower")

    ax[1].imshow(lbl_moy, cmap=cmap, origin="lower")

    for axis, title in zip(ax, titles):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title)

    return fig


########################################################
# SAVING ROUTINES
########################################################


def save_image_2d_cmd(image, file_name):
    split_name = file_name.split(os.sep)
    if len(split_name) == 1:
        data_file_path = "data" + os.sep + file_name
    else:
        data_file_path = (
            (os.sep).join(split_name[:-1]) + os.sep + "data" + os.sep + split_name[-1]
        )
    if image.shape > (1, 1):
        np.save(data_file_path, image)
        print(f"$ Image saved to disk: {data_file_path}.npy", "info")
    else:
        print("# Warning, image is empty", "Warning")


def save_image_as_blocks(img, full_filename, block_size_xy=256, label="raw_image"):
    num_planes = img.shape[0]
    block_size = (num_planes, block_size_xy, block_size_xy)
    blocks = view_as_blocks(img, block_shape=block_size).squeeze()
    print(f"\nDecomposing image into {blocks.shape[0] * blocks.shape[1]} blocks")

    folder = full_filename.split(".")[0]
    file_name = os.path.basename(full_filename).split(".")[0]

    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Folder created: {folder}")

    for i in trange(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            outfile = (
                folder
                + os.sep
                + file_name
                + "_"
                + label
                + "_block_"
                + str(i)
                + "_"
                + str(j)
                + ".tif"
            )
            imsave(outfile, blocks[i, j])


def image_show_with_values_single(
    ax, matrix, cbarlabel, fontsize, cbar_kw, valfmt="{x:.0f}", cmap="YlGn"
):
    row = [str(x) for x in range(matrix.shape[0])]
    im, _ = heatmap(
        matrix,
        row,
        row,
        ax=ax,
        cmap=cmap,
        cbarlabel=cbarlabel,
        fontsize=fontsize,
        cbar_kw=cbar_kw,
    )
    _ = annotate_heatmap(
        im, valfmt=valfmt, size=fontsize, threshold=None, textcolors=("black", "white")
    )  # , fontsize=fontsize


def image_show_with_values(
    matrices,
    output_name: str = "tmp.png",
    cbarlabels: list[str] = None,
    fontsize=6,
    verbose: bool = False,
    title="",
):
    """
    Plots a list of matrices with their values in each pixel.

    Parameters
    ----------
    matrices : list
        matrices to plot. Should be 2D numpy arrays
    output_name : TYPE, optional
        DESCRIPTION. The default is "tmp.png".
    cbarlabels : list, optional
        titles of subplots. The default is ["focalPlane"].
    fontsize : float, optional
        fontsize. The default is 6.
    verbose : Boolean, optional
        self explanatory. The default is False.
    title : str, optional
        figure title. The default is "".

    Returns
    -------
    None.

    """
    if cbarlabels is None:
        cbarlabels = ["focalPlane"]
    number_images = len(matrices)
    fig, axes = plt.subplots(1, number_images)
    fig.set_size_inches((number_images * 2, 5))
    fig.suptitle(title)
    cbar_kw = {"fraction": 0.046, "pad": 0.04}
    if len(cbarlabels) != number_images:
        cbarlabels = cbarlabels[0] * number_images

    if number_images == 1:
        ax = axes
        image_show_with_values_single(ax, matrices[0], cbarlabels[0], fontsize, cbar_kw)

    else:
        ax = axes.ravel()
        for matrix, axis, cbarlabel in zip(matrices, ax, cbarlabels):
            image_show_with_values_single(axis, matrix, cbarlabel, fontsize, cbar_kw)

    fig.tight_layout()
    plt.savefig(output_name)

    if not verbose:
        plt.close(fig)


def display_3d_assembled(
    images, localizations=None, plotting_range=None, normalize_b=True, masks=None
):
    wspace = 25

    rows_xy, cols_xy = images[0].shape[1], images[0].shape[0]
    rows_yz, cols_zx = images[2].shape[0], images[1].shape[0]
    rows = rows_xy + rows_yz + wspace
    cols = cols_xy + cols_zx + wspace

    fig = plt.figure(figsize=(50, 50), tight_layout=True)
    axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    display_image = np.zeros((rows, cols))
    display_image[0:rows_xy, 0:cols_xy] = images[0]
    display_image[rows_xy + wspace : rows_xy + rows_yz + wspace, 0:cols_xy] = images[2]
    display_image[0:rows_xy, cols_xy + wspace : cols_xy + cols_zx + wspace] = images[
        1
    ].transpose()

    if normalize_b:
        norm = simple_norm(display_image[:, :], "sqrt", percent=99)
        axis.imshow(display_image[:, :], cmap="Greys", alpha=1, norm=norm)
    else:
        axis.imshow(display_image[:, :], cmap="Greys", alpha=1)

    if masks is not None:
        display_mask = np.zeros((rows, cols))
        display_mask[0:rows_xy, 0:cols_xy] = masks[0]
        display_mask[rows_xy + wspace : rows_xy + rows_yz + wspace, 0:cols_xy] = masks[
            2
        ]
        display_mask[0:rows_xy, cols_xy + wspace : cols_xy + cols_zx + wspace] = masks[
            1
        ].transpose()

        axis.imshow(color.label2rgb(display_mask[:, :], bg_label=0), alpha=0.3)

    if localizations is not None:
        colors_rbgy = ["r", "g", "b", "y"]
        markersizes = [2, 1, 1, 1, 1]
        for i, loc in enumerate(localizations):
            axis.plot(loc[:, 2], loc[:, 1], "+", color=colors_rbgy[i], markersize=1)
            if plotting_range is not None:
                selections = [
                    np.abs(loc[:, a] - plotting_range[0]) < plotting_range[1]
                    for a in [2, 1]
                ]
                axis.plot(
                    loc[selections[0], 0] + cols_xy + wspace,
                    loc[selections[0], 1],
                    "+",
                    color=colors_rbgy[i],
                    markersize=markersizes[i],
                    alpha=0.9,
                )
                axis.plot(
                    loc[selections[1], 2],
                    loc[selections[1], 0] + rows_xy + wspace,
                    "+",
                    color=colors_rbgy[i],
                    markersize=markersizes[i],
                    alpha=0.9,
                )
            else:
                axis.plot(
                    loc[:, 0] + cols_xy,
                    loc[:, 1],
                    "+",
                    color=colors_rbgy[i],
                    markersize=1,
                )

    axis.axes.yaxis.set_visible(False)
    axis.axes.xaxis.set_visible(False)
    axis.axis("off")

    return fig


def _plot_image_3d(
    image_3d, localizations=None, masks=None, normalize_b=True, window=3
):
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
    img = image_3d
    center = int(img.shape[1] / 2)

    images = [np.sum(img, axis=0)]
    images.append(np.sum(img[:, :, center - window : center + window], axis=2))
    images.append(np.sum(img[:, center - window : center + window, :], axis=1))

    if masks is not None:
        labels = [np.max(masks, axis=0)]
        labels.append(np.max(masks[:, :, center - window : center + window], axis=2))
        labels.append(np.max(masks[:, center - window : center + window, :], axis=1))
    else:
        labels = None

    return display_3d_assembled(
        images,
        localizations=localizations,
        masks=labels,
        plotting_range=[center, window],
        normalize_b=normalize_b,
    )


def plot_3d_shift_matrices(shift_matrices, fontsize=8, log=False, valfmt="{x:.1f}"):
    cbar_kw = {"fraction": 0.046, "pad": 0.04}
    fig, axes = plt.subplots(1, len(shift_matrices))
    fig.set_size_inches((len(shift_matrices) * 10, 10))
    ax = axes.ravel()
    titles = ["z shift matrix", "x shift matrix", "y shift matrix"]

    for axis, title, x in zip(ax, titles, shift_matrices):
        if log:
            x = np.log10(x)
        image_show_with_values_single(
            axis, x, title, fontsize, cbar_kw, valfmt=valfmt, cmap="YlGn"
        )  # YlGnBu
        axis.set_title(title)

    return fig


def plot_4_images(allimages, titles=None):
    if titles is None:
        titles = [
            "reference",
            "cycle <i>",
            "processed reference",
            "processed cycle <i>",
        ]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    ax = axes.ravel()

    for axis, img, title in zip(ax, allimages, titles):
        axis.imshow(img, cmap="Greys")
        axis.set_title(title)
    fig.tight_layout()

    return fig


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    fontsize=12,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", size=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, size=fontsize)
    ax.set_yticklabels(row_labels, size=fontsize)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.1f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):  # sourcery skip: dict-assign-update-to-union
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    new_kwargs = dict(horizontalalignment="center", verticalalignment="center")
    new_kwargs.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_kwargs.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **new_kwargs)
            texts.append(text)

    return texts
