#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:00:35 2022

@author: marcnol

This class will handle correction of barcode positions from a table of local alignments

Remember that global alignments have already been corrected.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import glob
import os
import sys

import numpy as np
from astropy.table import Table
from tqdm import trange

from core.parameters import MatrixParams, RegistrationParams
from core.pyhim_logging import print_log, print_session_name, write_string_to_file
from imageProcessing.localization_table import LocalizationTable, decode_rois
from imageProcessing.makeProjections import Feature
from matrixOperations.filter_localizations import get_file_table_new_name


class RegisterLocalizationsTempo(Feature):
    def __init__(self, params: MatrixParams):
        super().__init__(params)
        self.out_folder = self.params.folder
        self.name = "RegisterLocalizations"


class RegisterLocalizations:
    def __init__(self, param, matrix_params: MatrixParams):
        """
        Parameters
        ----------
        param : class
            Parameters
        """

        self.current_param = param
        self.alignment_results_table_read = False
        self.found_match = []
        self.local_alignment_filename = None
        self.alignment_results_table = None
        self.dict_error_block_masks = None
        self.ndims = None

        self.tolerance_drift = matrix_params.toleranceDrift
        if isinstance(self.tolerance_drift, int):
            # defines a tuple suitable for anisotropic tolerance_drift (z,x,y)
            self.tolerance_drift = (
                self.tolerance_drift,
                self.tolerance_drift,
                self.tolerance_drift,
            )
        elif len(self.tolerance_drift) != 3:
            self.tolerance_drift = (
                3,
                1,
                1,
            )  # defines default anisotropic tolerance_drift (z,x,y)
        elif isinstance(self.tolerance_drift, list):
            self.tolerance_drift = tuple(self.tolerance_drift)

        self.remove_uncorrected_localizations = (
            matrix_params.remove_uncorrected_localizations
        )

        if self.remove_uncorrected_localizations:
            print_log("# Uncorrected localizations will be removed!!")
        else:
            print_log("# Uncorrected localizations will be kept!!")

    def search_local_shift(self, roi, barcode, zxy_uncorrected):
        if self.alignment_results_table_read:
            return self.search_local_shift_block_3d(roi, barcode, zxy_uncorrected)
        print("ERROR> did not found alignment_results_table")
        return zxy_uncorrected, {"below_tolerance": False}

    def search_local_shift_block_3d(self, roi, barcode, zxy_uncorrected):
        """
        Searches for local drift for a specific barcode in a given roi.
        If it exists then it adds to the uncorrected coordinates

        Parameters
        ----------
        roi : int
            roi used
        x_uncorrected : float
            x coordinate.
        y_uncorrected : float
            y coordinate.

        Returns
        -------
        x_corrected : float
            corrected x coordinate.
        y_corrected : float
            corrected y coordinate.

        """

        # gets blockSize
        block_size_xy = self.alignment_results_table[0]["blockXY"]

        # zxy coord in block reference coord system
        zxy_block = [np.floor(a / block_size_xy).astype(int) for a in zxy_uncorrected]

        # defines roi, barcode ID, and blocks for the localization
        n_roi = f"ROI:{str(roi)}"
        n_barcode = "barcode:" + "RT" + str(barcode)
        n_block_i = f"block_i:{str(zxy_block[1])}"
        n_block_j = f"block_j:{str(zxy_block[2])}"

        # finds the corresponding shift int the dictionary
        shifts = [
            self.dict_error_block_masks[n_roi][n_barcode][n_block_i][n_block_j][
                "shift_z"
            ],
            self.dict_error_block_masks[n_roi][n_barcode][n_block_i][n_block_j][
                "shift_x"
            ],
            self.dict_error_block_masks[n_roi][n_barcode][n_block_i][n_block_j][
                "shift_y"
            ],
        ]

        accepts_localization = False
        if isinstance(self.tolerance_drift, tuple):
            # makes list with comparisons per axis
            check = [
                np.abs(shift) < tol for shift, tol in zip(shifts, self.tolerance_drift)
            ]
            if all(check):
                accepts_localization = True  # only if tolerance is passed in all axes the localization is kept
        # defaults to previous usage with isotropic tolerance
        elif max(np.abs(shifts)) < self.tolerance_drift:
            accepts_localization = True

        if accepts_localization:
            zxy_corrected = [a + shift for a, shift in zip(zxy_uncorrected, shifts)]
            quality_correction = {"below_tolerance": True}
        else:
            zxy_corrected = zxy_uncorrected
            quality_correction = {"below_tolerance": False}

        return zxy_corrected, quality_correction

    def register_barcodes(self, barcode_map, reg_params: RegistrationParams):
        """
        This function will take a barcode_map and a Table of 3D alignments to register barcode coordinates

        Returns
        -------
        None.

        """

        reference_fiducial = reg_params.referenceFiducial
        block_size = reg_params.blockSize
        print_log(
            f"\n$ Parameters:\n Blocksize = {block_size}\n Tolerance = {self.tolerance_drift}\n Reference barcode = {reference_fiducial}"
        )

        list_uncorrected_barcodes = []

        # loops over barcode Table rows in a given roi
        for i in trange(
            len(barcode_map.groups[0])
        ):  # i is the index of the barcode in barcode_map_roi
            barcode = barcode_map.groups[0]["Barcode #"][i]
            roi = barcode_map.groups[0]["ROI #"][i]

            # keeps the particle if the test passed
            x_uncorrected = barcode_map.groups[0]["ycentroid"][i]
            y_uncorrected = barcode_map.groups[0]["xcentroid"][i]
            z_uncorrected = barcode_map.groups[0]["zcentroid"][i]

            # TODO: UPDATE this comment, localDriftCorrection was removed from pyHiM.
            # Corrects XYZ coordinate of barcode if localDriftCorrection is available
            zxy_uncorrected = [z_uncorrected, x_uncorrected, y_uncorrected]
            rt_barcode = f"RT{str(barcode)}"

            if rt_barcode != reg_params.referenceFiducial:
                zxy_corrected, quality_correction = self.search_local_shift(
                    roi, barcode, zxy_uncorrected
                )
                if not quality_correction["below_tolerance"]:
                    list_uncorrected_barcodes.append(i)
                    if self.remove_uncorrected_localizations:
                        # will remove localizations that cannot be corrected
                        zxy_corrected = [np.nan, np.nan, np.nan]

                    # ELSE: will keep uncorrected localizations

            else:
                # if it is the reference cycle, then it does not correct coordinates
                zxy_corrected = zxy_uncorrected

            # rewrites corrected XYZ values to Table
            # if not quality_correction['below_tolerance'] and self.remove_uncorrected_localizations:
            # print(f" $ Before correction: {barcodeMap.groups[0][i]} ")

            barcode_map.groups[0]["ycentroid"][i] = zxy_corrected[1]
            barcode_map.groups[0]["xcentroid"][i] = zxy_corrected[2]
            if self.ndims > 2:
                barcode_map.groups[0]["zcentroid"][i] = zxy_corrected[0]

                # if not quality_correction['below_tolerance'] and self.remove_uncorrected_localizations:
                # print(f" $ After correction: {barcodeMap.groups[0][i]} ")

        if self.remove_uncorrected_localizations:
            nb_loc_before = len(barcode_map.groups[0])
            # Remove rows from the end
            for i in list_uncorrected_barcodes[::-1]:
                barcode_map.remove_row(i)
            print_log(
                f"$ {len(list_uncorrected_barcodes)} localizations out of {nb_loc_before} were removed."
            )
        else:
            print_log(
                f"$ {len(list_uncorrected_barcodes)} localizations out of {len(barcode_map.groups[0])} were uncorrected."
            )

        return barcode_map

    def load_local_alignment(self, local_shifts_path, reg_params: RegistrationParams):
        if reg_params.localAlignment != "None":
            return self._load_local_alignment(local_shifts_path, reg_params)
        print_log("\n\n$ localAlignment option set to `None`")
        return False, Table()

    def _load_local_alignment(self, local_shifts_path, reg_params: RegistrationParams):
        mode = reg_params.localAlignment
        self.local_alignment_filename = local_shifts_path

        if os.path.exists(self.local_alignment_filename):
            self.alignment_results_table = Table.read(
                self.local_alignment_filename, format="ascii.ecsv"
            )
            self.alignment_results_table_read = True

            # builds dict of local alignments
            self.build_local_alignment_dict()

            print_log(f"$ LocalAlignment file loaded: {self.local_alignment_filename}")
            print_log(f"$ Will correct coordinates using {mode} alignment")
            print_log(f"$ Number of records: {len(self.alignment_results_table)}")
        else:
            print_log(
                f"\n\n# Warning: could not find localAlignment: {self.local_alignment_filename}"
            )
            print_log("\tProceeding with only global alignments...")

            self.alignment_results_table_read = False
            self.alignment_results_table = Table()
            self.dict_error_block_masks = Table()

    def build_local_alignment_dict(self):
        """
        Builds dictionary of local corrections for each ROI, barcode cycle, and block combination

        Parameters
        ----------
        self.alignment_results_table: astropy Table
            alignment_results_table table

        self.alignment_results_table_read: Boolean
            True when alignment_results_table table was read from disk

        Returns
        -------
        exit_code: Boolean

        self.dict_error_block_masks: dict

        """
        if not self.alignment_results_table_read:
            print_log("Did not find alignment_results_table. Cannot continue")
            return False
        else:
            alignment_results_table = self.alignment_results_table

        dict_error_block_masks = {}

        for row in alignment_results_table:
            n_roi = "ROI:" + str(row["ROI #"])
            n_barcode = "barcode:" + row["label"]
            n_block_i = "block_i:" + str(row["block_i"])
            n_block_j = "block_j:" + str(row["block_j"])

            if n_roi not in dict_error_block_masks:
                dict_error_block_masks[n_roi] = {}

            if n_barcode not in dict_error_block_masks[n_roi].keys():
                dict_error_block_masks[n_roi][n_barcode] = {}

            if n_block_i not in dict_error_block_masks[n_roi][n_barcode].keys():
                dict_error_block_masks[n_roi][n_barcode][n_block_i] = {}

            if (
                n_block_j
                not in dict_error_block_masks[n_roi][n_barcode][n_block_i].keys()
            ):
                dict_error_block_masks[n_roi][n_barcode][n_block_i][n_block_j] = {}

            dict_error_block_masks[n_roi][n_barcode][n_block_i][n_block_j] = {
                "shift_z": row["shift_z"],
                "shift_x": row["shift_x"],
                "shift_y": row["shift_y"],
                "quality_xy": row["quality_xy"],
                "quality_zy": row["quality_zy"],
                "quality_zx": row["quality_zx"],
            }

        self.dict_error_block_masks = dict_error_block_masks
        return True

    def register_barcode_map_file(self, file, reg_params: RegistrationParams):
        self.ndims = 3 if "3D" in os.path.basename(file) else 2
        # loads barcode coordinate Tables
        table = LocalizationTable()

        barcode_map_full, _ = table.load(
            file
        )  # barcode_map_full, unique_barcodes = table.load(file)

        # checks that everything is OK
        if len(barcode_map_full) < 1:
            print_log(f"\nWARNING>{file} contains an empty table!")
            return None

        if (
            "comments" in barcode_map_full.meta.keys()
            and "registered" in barcode_map_full.meta["comments"]
        ):
            print_log(
                f"\nWARNING>{file} contains a table thas was already registered! \nWill not do anything"
            )
            return None

        # preserves original copy of table for safe keeping
        new_file = get_file_table_new_name(file)
        table.save(new_file, barcode_map_full)
        barcode_map_full_unregistered = barcode_map_full.copy()

        # indexes table by ROI
        barcode_map_roi, number_rois = decode_rois(barcode_map_full)

        for i_roi in range(number_rois):
            # creates sub Table for this ROI
            barcode_map = barcode_map_roi.groups[i_roi]
            n_roi = barcode_map["ROI #"][0]
            print_log(f"\nProcessing barcode localization Table for ROI: {n_roi}")

            # registers localizations
            barcode_map = self.register_barcodes(barcode_map, reg_params)

        # saves and plots registered barcode coordinate Tables
        table.save(file, barcode_map, comments="registered")
        filepath_split = file.split(".")[0].split(os.sep)  # remove ext + split path
        filepath_split.remove("data")
        filepath_without_data_folder = (os.sep).join(filepath_split)
        table.plot_distribution_fluxes(
            barcode_map, [filepath_without_data_folder, "_registered", "_stats", ".png"]
        )
        table.plots_localizations(
            barcode_map,
            [filepath_without_data_folder, "_registered", "", ".png"],
        )
        table.compares_localizations(
            barcode_map,
            barcode_map_full_unregistered,
            [filepath_without_data_folder, "_registered", "_comparisons", ".png"],
        )

    def register(
        self, data_path, local_shifts_path, seg_params, reg_params: RegistrationParams
    ):
        """
        Function that registers barcodes using a local drift correction table produced by *register_local*


        Returns
        -------
        None.

        """
        session_name = "register_localizations"

        # processes folders and files
        print_session_name(session_name)
        write_string_to_file(
            self.current_param.param_dict["fileNameMD"],
            f"## {session_name}\n",
            "a",
        )
        label = "barcode"

        current_folder = data_path
        print_log(f"> Processing Folder: {current_folder}")

        # Loads localAlignment if it exists otherwise it exits with error
        self.load_local_alignment(local_shifts_path, reg_params)

        if not self.alignment_results_table_read:
            print_log(
                "Unable to find alignment table.\nDid you run register_local?\n\n "
            )
            sys.exit(
                f"ERROR: Expected to find: {self.local_alignment_filename}--> Aborting."
            )

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
        files = list(glob.glob(data_file_base_2d + "_*" + label + ".dat"))
        files += list(glob.glob(data_file_base_3d + "_*" + label + ".dat"))

        if not files:
            print_log("No localization table found to process!")
            return

        for file in files:
            self.register_barcode_map_file(file, reg_params)

        print_log(f" {len(files)} barcode tables processed in {current_folder}")
