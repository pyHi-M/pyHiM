#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:06:14 2022

@author: marcnol

This class will contain methods to load, save, plot barcode localizations and statistics

"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from apifish.stack.io import read_table_from_ecsv
from astropy.table import vstack

from core.pyhim_logging import print_log


class LocalizationTable:
    def __init__(self):
        self.a = 1

    def load(self, file):
        """
        Loads barcode_map

        Parameters
        ----------
        filename_barcode_coordinates : string
            filename with barcode_map

        Returns
        -------
        barcode_map : Table()
        unique_barcodes: list
            lis of unique barcodes read from barcode_map

        """
        if os.path.exists(file):
            # barcode_map = Table.read(filename_barcode_coordinates, format="ascii.ecsv")
            barcode_map = read_table_from_ecsv(file)

            print_log(f"$ Successfully loaded barcode localizations file: {file}")

            unique_barcodes = np.unique(barcode_map["Barcode #"].data)
            number_unique_barcodes = unique_barcodes.shape[0]

            print_log(
                f"$ Number of barcodes read from barcode_map: {number_unique_barcodes}"
            )
            print_log(f"$ Unique Barcodes detected: {unique_barcodes}")
        else:
            print_log(
                f"\n\n# ERROR: could not find coordinates file: {file}", status="DEBUG"
            )
            sys.exit()

        return barcode_map, unique_barcodes

    def save(self, file_name, barcode_map, comments=""):
        """
        Saves output table

        Parameters
        ----------
        filename_barcode_coordinates : string
            filename of table.
        barcode_map : astropy Table
            Table to be written to file.
        tag : string, optional
            tag to be added to filename. The default is "_".
        ext : string, optional
            file extension. The default is 'ecsv'.
        comments : list of strings, optional
            Will output as comments to the header. The default is [].

        Returns
        -------
        None.

        """
        print_log(f"$ Saving output table as {file_name} ...")

        try:
            barcode_map.meta["comments"].append(comments)
        except KeyError:
            barcode_map.meta["comments"] = [comments]

        barcode_map.write(
            file_name,
            format="ascii.ecsv",
            overwrite=True,
        )

    def append(self, table1, table2):
        """
        appends <table> to self.data

        Parameters
        ----------
        table : astropy table
            table to append to existing self.data table.

        Returns
        -------
        None.

        """

        return vstack([table1, table2])

    def plot_distribution_fluxes(self, barcode_map, filename_list):
        """
        This function plots the distribution of fluxes, sharpness, roundness, magnitude and peak intensity from a Table

        Parameters
        ----------
        barcode_map : TYPE
            DESCRIPTION.
        filename_list: list
            filename

        Returns
        -------
        None.

        """

        # initializes figure
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        fig.set_size_inches((10, 5))

        # initializes variables
        roundness = barcode_map["roundness1"]
        peak = barcode_map["peak"]
        zcentroid = barcode_map["zcentroid"]
        flux = barcode_map["flux"]

        # plots data
        ax[0].scatter(peak, zcentroid, c=peak, cmap="Reds", alpha=0.5)
        ax[0].set_title("color: peak intensity")
        ax[0].set_ylabel("zcentroid")
        ax[0].set_xlabel("peak intensity")

        p_2 = ax[1].scatter(roundness, flux, c=peak, cmap="Reds", alpha=0.5)
        ax[1].set_title("color: peak intensity")
        ax[1].set_xlabel("roundness")
        ax[1].set_ylabel("flux")
        fig.colorbar(p_2, ax=ax[1], fraction=0.046, pad=0.04)

        # saves figure
        fig.savefig("".join(filename_list))

        plt.close(fig)

    def plot_intensity_distribution(
        self, intensities, output_file="intensity_distribution.png"
    ):
        """
        Plots the distribution of intensity values from the localization table
        with a logarithmic y-axis and saves it as a PNG file.
        """

        plt.figure(figsize=(8, 6))
        plt.hist(intensities, bins=50, color="blue", alpha=0.7, edgecolor="black")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency (log scale)")
        plt.title("Localization Intensity Distribution")
        plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"$ Saved intensity distribution plot as {output_file}")

    def plots_localizations(self, barcode_map_full, filename_list):
        """
        This function plots 3 subplots (xy, xz, yz) with the localizations.
        One figure is produced per ROI.

        Parameters
        ----------
        image : List of numpy ndarray (N-dimensional array)
            3D raw image of format .tif

        label : List of numpy ndarray (N-dimensional array)
            3D labeled image of format .tif

        filename_list: list
            filename
        """

        # indexes table by ROI
        barcode_map_roi, number_rois = decode_rois(barcode_map_full)

        for i_roi in range(number_rois):
            # creates sub Table for this ROI
            barcode_map = barcode_map_roi.groups[i_roi]
            n_roi = barcode_map["ROI #"][0]
            print_log(f"> Plotting barcode localization map for ROI: {n_roi}")
            color_dict = build_color_dict(barcode_map, key="Barcode #")

            # initializes figure
            fig = plt.figure(constrained_layout=False)
            im_size = 60
            fig.set_size_inches((im_size * 2, im_size))
            gs = fig.add_gridspec(2, 2)
            ax = [
                fig.add_subplot(gs[:, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 1]),
            ]

            # defines variables
            x = barcode_map["xcentroid"]
            y = barcode_map["ycentroid"]
            z = barcode_map["zcentroid"]
            colors = [color_dict[str(x)] for x in barcode_map["Barcode #"]]
            titles = ["Z-projection", "X-projection", "Y-projection"]

            # makes plot
            plots_localization_projection(x, y, ax[0], colors, titles[0])
            plots_localization_projection(x, z, ax[1], colors, titles[1])
            plots_localization_projection(y, z, ax[2], colors, titles[2])

            fig.tight_layout()

            # saves output figure
            filename_list_i = filename_list.copy()
            filename_list_i.insert(-1, f"_ROI{str(n_roi)}")
            fig.savefig("".join(filename_list_i))

    def compares_localizations(
        self, barcode_map_1, barcode_map_2, filename_list, fontsize=20
    ):
        """
        Compares the localizations of two barcode tables

        Parameters
        ----------
        barcode_map_1 : astropy Table
            localization table 1.
        barcode_map_2 : astropy Table
            localization table 2.

        Returns
        -------
        None.

        """

        barcode_map_2.add_index("Buid")
        number_localizations = len(barcode_map_1)

        labels = ["xcentroid", "ycentroid", "zcentroid"]
        diffs = {label: [] for label in labels}
        # iterates over rows in barcode_map_1
        for row in range(number_localizations):
            buid_1 = barcode_map_2[row]["Buid"]
            barcode_found = True

            # finds same Buid in barcode_map_2
            try:
                barcode_map_2.loc[buid_1]
            except KeyError:
                barcode_found = False

            # collects differences in values between same localization in both tables
            if barcode_found:
                for label in labels:
                    a, b = barcode_map_2.loc[buid_1][label], barcode_map_1[row][label]
                    if ~np.isnan(a).any() and ~np.isnan(b).any():
                        # diff = a - b
                        # if np.isnan(diff):
                        #    diff = 0
                        diffs[label].append(a - b)

        # plots figures
        fig, axes = plt.subplots(2, 2)
        ax = axes.ravel()
        fig.set_size_inches((30, 30))

        for label, axis in zip(labels, ax):
            r = np.array(diffs[label])
            axis.hist(r, bins=20)
            axis.set_xlabel(f"{label} correction, px", fontsize=fontsize)
            axis.set_ylabel("counts", fontsize=fontsize)

        ax[3].scatter(
            np.array(diffs["ycentroid"]), np.array(diffs["xcentroid"]), s=3, alpha=0.8
        )
        ax[3].set_xlabel("dx-position, px", fontsize=fontsize)
        ax[3].set_ylabel("dy-position, px", fontsize=fontsize)

        fig.savefig("".join(filename_list))


def decode_rois(data):
    data_indexed = data.group_by("ROI #")

    number_rois = len(data_indexed.groups.keys)

    print_log(f"\n$ rois detected: {number_rois}")

    return data_indexed, number_rois


def build_color_dict(data, key="Barcode #"):
    unique_barcodes = np.unique(data[key])
    output_array = range(unique_barcodes.shape[0])

    return {
        str(barcode): output for barcode, output in zip(unique_barcodes, output_array)
    }


def plots_localization_projection(coord1, coord2, axis, colors, title="" * 3):
    """
    This function will produce the scatter plot and add title

    Parameters
    ----------
    coord1 : 1D Numpy array, float
        first coordinate (x).
    coord2 : 1D Numpy array, float
        first coordinate (y).
    axis : matplotlib axis
        figure axis handle.
    colors : 1D Numpy array, float
        colorcode used in scatter plot.
    title : string, optional
        title of subpanel. The default is ''*3.

    Returns
    -------
    None.

    """
    axis.scatter(coord1, coord2, s=5, c=colors, alpha=0.9, cmap="hsv")  # nipy_spectral
    axis.set_title(title)
