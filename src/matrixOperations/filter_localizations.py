#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:45:44 2022

@author: marcnol
"""
# =============================================================================
# IMPORTS
# =============================================================================

import glob
import os

from tqdm import trange

from core.parameters import MatrixParams
from core.pyhim_logging import print_log, print_session_name, write_string_to_file
from imageProcessing.localization_table import LocalizationTable
from imageProcessing.makeProjections import Feature


class FilterLocalizationsTempo(Feature):
    def __init__(self, params: MatrixParams):
        super().__init__(params)
        self.out_folder = self.params.folder
        self.name = "FilterLocalizations"


class FilterLocalizations:
    def __init__(self, param):
        """
        Parameters
        ----------
        param : class
            Parameters
        """

        self.current_param = param

    def filter_localizations_quality(self, barcode_map, i):
        """
        [filters barcode localizations either by brightness or 3D localization accuracy]

        Parameters
        ----------
        i : int
            index in barcode_map Table


        Returns
        -------
        keep : Boolean
            True if the test is passed.

        """
        return barcode_map["flux"][i] > self.flux_min

    def filter_barcode_table(self, barcode_map):
        """
        iterates over rows of a barcode localization table and filters unwanted rows

        Parameters
        ----------
        barcode_map_roi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        rows_to_remove = []
        n_barcodes = len(barcode_map)
        print_log(f"$ Minimum flux: {self.flux_min}")
        # [filters barcode per blockAlignmentMask, if existing]
        # keep_alignment = self.filter_localizations_block_alignment(barcode_map, i)
        keep_alignment = True

        for i in trange(n_barcodes):  # i is the index of the barcode in barcode_map_roi
            # [filters barcode localizations either by]
            keep_quality = self.filter_localizations_quality(barcode_map, i)

            if not keep_quality or not keep_alignment:
                rows_to_remove.append(i)

        # removes rows from table
        barcode_map.remove_rows(rows_to_remove)

        print_log(
            f"$ Removed {len(rows_to_remove)} barcode localizations from table out of {n_barcodes}."
        )

        return barcode_map

    def setup_filter_values(self, matrix_params: MatrixParams):
        """
        Returns
        -------
        self.block_size : int
            size of blocks used for blockAlignment.
        self.flux_min : float
            Minimum flux to keep barcode localization

        """
        if self.ndims == 3:
            self.flux_min = matrix_params.flux_min_3D
        else:
            self.flux_min = matrix_params.flux_min

    def filter_folder(
        self,
        data_path,
        seg_params,
        matrix_params: MatrixParams,
    ):
        """
        Function that filters barcodes using a number of user-provided parameters


        Returns
        -------
        None.

        """
        session_name = "filter_localizations"

        # processes folders and files
        print_session_name(session_name)
        write_string_to_file(
            self.current_param.param_dict["fileNameMD"],
            f"## {session_name}\n",
            "a",
        )

        current_folder = data_path
        print_log(f"> Processing Folder: {current_folder}")

        data_file_base_2d = (
            data_path
            + os.sep
            + seg_params.localize_2d_folder
            + os.sep
            + "data"
            + os.sep
            + seg_params.outputFile
        )
        data_file_base_3d = (
            data_path
            + os.sep
            + seg_params.localize_3d_folder
            + os.sep
            + "data"
            + os.sep
            + seg_params.outputFile
        )
        files = list(glob.glob(data_file_base_2d + "_*barcode.dat"))
        files += list(glob.glob(data_file_base_3d + "_*barcode.dat"))
        if files:
            for file in files:
                self.ndims = 3 if "3D" in os.path.basename(file) else 2
                self.setup_filter_values(matrix_params)

                # Loads barcode coordinate Tables
                table = LocalizationTable()
                barcode_map, _ = table.load(file)

                if len(barcode_map) > 0:
                    # plots and saves original barcode coordinate Tables for safe keeping
                    new_file = get_file_table_new_name(file)
                    table.save(new_file, barcode_map)
                    # remove ext + split path
                    filepath_split = new_file.split(".")[0].split(os.sep)
                    filepath_split.remove("data")
                    filepath_without_data_folder = (os.sep).join(filepath_split)
                    table.plot_distribution_fluxes(
                        barcode_map,
                        [filepath_without_data_folder, "_stats", ".png"],
                    )
                    table.plots_localizations(
                        barcode_map,
                        [filepath_without_data_folder, "", ".png"],
                    )

                    # processes tables
                    barcode_map_roi = barcode_map.group_by("ROI #")
                    number_rois = len(barcode_map_roi.groups.keys)
                    print_log(f"\n$ rois detected: {number_rois}")

                    # Filters barcode coordinate Tables
                    barcode_map = self.filter_barcode_table(barcode_map)

                    # saves and plots filtered barcode coordinate Tables
                    table.save(file, barcode_map, comments="filtered")
                    filepath_split = file.split(".")[0].split(
                        os.sep
                    )  # remove ext + split path
                    filepath_split.remove("data")
                    filepath_without_data_folder = (os.sep).join(filepath_split)
                    table.plot_distribution_fluxes(
                        barcode_map, [filepath_without_data_folder, "_stats", ".png"]
                    )
                    table.plots_localizations(
                        barcode_map,
                        [filepath_without_data_folder, "", ".png"],
                    )

                else:
                    print_log(f"\nWARNING>{file} contains an empty table!")

        else:
            print_log("No barcode tables found!")

        print_log(f"Barcode tables {current_folder} filtered", "info")


def get_file_table_new_name(file):
    existing_versions = glob.glob(file.split(".")[0] + "_version_*.dat")

    if len(existing_versions) < 1:
        new_version = 0
    else:
        version_numbers = [
            int(x.split("_version_")[1].split(".")[0]) for x in existing_versions
        ]

        new_version = max(version_numbers) + 1 if version_numbers else 0
    return file.split(".dat")[0] + "_version_" + str(new_version) + ".dat"
