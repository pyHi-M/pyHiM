#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script will build chromatin traces using a segmentObjects_barcode table.

The methods that will be implemented are:

- 1 = assignment by mask (either DAPI mask or other)
- 2 = spatial clusterization using KDtree. This method is mask-free.

Method 1 iterates over rois:

- assigns barcode localizations to masks
- applies local drift correction, if available
- removes localizations using flux and driftTolerance
- calculates the pair-wise distances for each single-cell mask

Outputs are:

- Table with #cell #PWD #coordinates (e.g. buildsPWDmatrix_3D_order:0_ROI:1.ecsv)
- NPY array with single cell PWD single cell matrices (e.g. buildsPWDmatrix_3D_HiMscMatrix.npy)
- NPY array with barcode identities (e.g. buildsPWDmatrix_3D_uniqueBarcodes.ecsv)
- the files with no "3D" tag contain data analyzed using 2D localizations.

Single-cell results are combined together to calculate:

- Distribution of pairwise distance for each barcode combination
- Ensemble mean pairwise distance matrix using mean of distribution
- Ensemble mean pairwise distance matrix using Kernel density estimation
- Ensemble Hi-M matrix using a predefined threshold
- For each of these files, there is an image in PNG format saved. Images containing "3D" are for 3D other are for 2D.

"""


import glob
import os
import uuid

import numpy as np
from apifish.stack.io import read_array
from scipy.spatial import KDTree
from skimage.measure import label
from skimage.segmentation import expand_labels
from tqdm import trange
from tqdm.contrib import tzip

from core.parameters import AcquisitionParams, MatrixParams
from core.pyhim_logging import print_dashes, print_log
from imageProcessing.localization_table import LocalizationTable
from imageProcessing.makeProjections import Feature
from matrixOperations.chromatin_trace_table import ChromatinTraceTable


class BuildTracesTempo(Feature):
    def __init__(self, params: MatrixParams):
        super().__init__(params)
        self.out_folder = self.params.folder
        self.name = "BuildTraces"


class BuildTraces:
    def __init__(self, param, acq_params: AcquisitionParams):
        self.current_param = param

        self.initialize_parameters(acq_params)

        # initialize with default values
        self.current_folder = []
        self.mask_identifier = ["DAPI"]  # default mask label
        self.masks = np.zeros((2048, 2048))

    def initializes_masks(self, masks):
        self.masks = masks
        self.n_cells_assigned = 0
        self.n_cells_unassigned = 0
        self.n_barcodes_in_mask = 0
        self.number_masks = np.max(self.masks).astype(int)
        self.barcodes_in_mask = {
            f"maskID_{str(mask)}": [] for mask in range(self.number_masks + 1)
        }

    def initialize_parameters(self, acq_params: AcquisitionParams):
        # initializes parameters from current_param

        self.z_binning = acq_params.zBinning
        self.pixel_size_xy = acq_params.pixelSizeXY
        self.pixel_size_z_0 = acq_params.pixelSizeZ
        self.pixel_size_z = self.z_binning * self.pixel_size_z_0
        self.log_name_md = self.current_param.param_dict["fileNameMD"]

    def initialize_lists(self):
        (
            self.rois,
            self.cell_id,
            self.n_barcodes,
            self.barcode_ids,
            self.cuid,
            self.buid,
            self.barcode_coordinates,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    def align_by_masking(self, matrix_params: MatrixParams):
        """
        Assigns barcodes to masks and creates <n_barcodes_in_mask>
        And by filling in the "Cell #" key of barcode_map_roi
        This routine will only select which barcodes go to each cell mask

        Returns
        -------
        self.barcodes_in_mask # dictionary with the identities of barcodes contained in each mask.
            Keys: 'maskID_1', 'maskID_2', and so on

        self.n_barcodes_in_mask # vector containing the number of barcodes for each mask
        self.n_cells_assigned # number of cells assigned
        self.n_cells_unassigned # number of cells unassigned
        """

        n_barcodes_in_mask = np.zeros(self.number_masks + 2)
        n_barcodes_roi = 0

        image_size_array = self.masks.shape
        if len(image_size_array) == 2:
            # 2d
            image_size = {
                "z": np.inf,
                "x": image_size_array[0],
                "y": image_size_array[1],
            }
        else:
            # 3d
            image_size = {
                "z": image_size_array[0],
                "x": image_size_array[1],
                "y": image_size_array[2],
            }

        # loops over barcode Table rows in a given ROI
        print_log(f"> Aligning localizations to {self.number_masks} masks...")
        for i in trange(
            len(self.barcode_map_roi.groups[0])
        ):  # i is the index of the barcode in barcode_map_roi
            # gets xyz coordinates
            x_corrected = self.barcode_map_roi.groups[0]["ycentroid"][i]
            y_corrected = self.barcode_map_roi.groups[0]["xcentroid"][i]

            if self.ndims == 2:
                z_corrected = self.barcode_map_roi.groups[0]["zcentroid"][i] = 0.0
            else:
                z_corrected = self.barcode_map_roi.groups[0]["zcentroid"][i]

            # binarizes coordinate
            y_int = binarize_coordinate(y_corrected)
            x_int = binarize_coordinate(x_corrected)
            z_int = binarize_coordinate(z_corrected) + int(matrix_params.z_offset)

            # finds what mask label this barcode is sitting on
            if np.isnan(x_int) or np.isnan(y_int) or np.isnan(z_int):
                # if a barcode has coordinates with NaNs, it is assigned to background
                mask_id = 0
            elif (
                x_int < image_size["x"]
                and y_int < image_size["y"]
                and z_int < image_size["z"]
                and x_int > 0
                and y_int > 0
                and z_int > 0
            ):
                mask_id = (
                    self.masks[x_int, y_int]
                    if len(image_size_array) == 2
                    else self.masks[z_int, x_int, y_int]
                )

            else:
                # if a barcode has coordinates outside the image, it is assigned to background
                mask_id = 0

            # attributes CellID to a barcode
            self.barcode_map_roi["CellID #"][i] = mask_id

            # if it is not background,
            if mask_id > 0:
                # increments counter of number of barcodes in the cell mask attributed
                n_barcodes_in_mask[mask_id] += 1

                # stores the identify of the barcode in a mask dictionary
                self.barcodes_in_mask[f"maskID_{str(mask_id)}"].append(i)

                # keeps statistics
                if int(self.barcode_map_roi.groups[0]["ROI #"][i]) == int(self.n_roi):
                    n_barcodes_roi += 1

        # Total number of masks assigned and not assigned
        self.n_cells_assigned = np.count_nonzero(n_barcodes_in_mask > 0)
        self.n_cells_unassigned = self.number_masks - self.n_cells_assigned

        # this list contains which barcodes are allocated to which masks
        self.n_barcodes_in_mask = n_barcodes_in_mask

        print_log(
            f"$ Number of cells assigned: {self.n_cells_assigned} \
                | discarded: {self.n_cells_unassigned}"
        )

    def build_vector(self, x, y, z):
        """
        Builds vector from coordinates

        Parameters
        ----------
        x : float
            x coordinates
        y : float
            y coordinates
        z : float
            z coordinates

        Returns
        -------
        coords : np array
            vector with coordinates in nanometers.

        """

        return np.column_stack(
            (
                x * self.pixel_size["x"],
                y * self.pixel_size["y"],
                z * self.pixel_size["z"],
            )
        )

    def builds_sc_distance_table(self):
        """
        iterates over all masks, calculates PWD for each mask, assigns them to sc_distance_table

        Returns
        -------
        sc_distance_table

        """
        # sorts Table by cellID
        barcode_map_roi = self.barcode_map_roi

        # indexes table by cellID
        barcode_map_roi_cell_id = barcode_map_roi.group_by("CellID #")

        self.initialize_lists()

        # iterates over all traces in an ROI
        print_log("> Building single traces")
        for key, group in tzip(
            barcode_map_roi_cell_id.groups.keys, barcode_map_roi_cell_id.groups
        ):
            if key["CellID #"] > 1:  # excludes trace 0 as this is background
                cell_id, roi = (key["CellID #"], group["ROI #"].data[0])
                trace_id = str(uuid.uuid4())

                # gets lists of x, y and z coordinates for barcodes assigned to a cell mask
                x, y, z = (
                    np.array(group["xcentroid"].data),
                    np.array(group["ycentroid"].data),
                    np.array(group["zcentroid"].data),
                )

                # gets vector in nm
                r_mum = self.build_vector(x, y, z)

                for i in range(x.shape[0]):
                    entry = [
                        group["Buid"].data[i],  # spot uid
                        trace_id,  # trace uid
                        r_mum[i][0],  # x, microns
                        r_mum[i][1],  # y, microns
                        r_mum[i][2],  # z, microns
                        "xxxxx",  # chromosome
                        0,  # start sequence
                        999999999,  # end sequence
                        roi,  # ROI number
                        cell_id,  # Mask number
                        group["Barcode #"].data[i],  # Barcode name
                        "x" * 20,  # label
                    ]
                    self.trace_table.data.add_row(entry)

        print_log(f"$ Coordinates dimensions: {self.ndims}")

    def load_mask(
        self,
        files_in_folder,
        data_path,
        seg_params,
        acq_params: AcquisitionParams,
        matrix_params: MatrixParams,
    ):  # sourcery skip: use-named-expression
        """
        searches and loads mask files for building chromatin trace

        Parameters
        ----------
        files_in_folder : list of str
            list of TIF files to be explored.

        Returns
        -------
        bool
            True: mask found and loaded
            False: failed to find mask file

        """

        # finds files with cell masks
        channel = getattr(acq_params, f"{self.mask_type}_channel")

        files_to_process = [
            file
            for file in files_in_folder
            if self.current_param.decode_file_parts(file)["channel"]
            == channel  # typically "ch00"
            and self.mask_identifier.split("_")[0] in os.path.basename(file).split("_")
            and int(self.current_param.decode_file_parts(file)["roi"]) == self.n_roi
        ]

        if files_to_process:
            # loads file with cell masks
            filename_masks_2d = (
                os.path.basename(files_to_process[0]).split(".")[0] + "_Masks.npy"
            )
            filename_masks_3d = (
                os.path.basename(files_to_process[0]).split(".")[0] + "_3Dmasks.npy"
            )
            # TODO: Check if we don't forget to allow a 3D masks loading ?
            full_filename_masks_2d = (
                data_path
                + os.sep
                + seg_params.mask_2d_folder
                + os.sep
                + "data"
                + os.sep
                + filename_masks_2d
            )
            full_filename_masks_3d = (
                data_path
                + os.sep
                + seg_params.mask_3d_folder
                + os.sep
                + "data"
                + os.sep
                + filename_masks_3d
            )

            if "3D" in self.mask_identifier.split("_"):
                full_filename_masks = full_filename_masks_3d
                print_log("3D masks used !")
            elif "2D" in self.mask_identifier.split("_"):
                full_filename_masks = full_filename_masks_2d
                print_log("2D masks used !")
            elif self.ndims == 3 and os.path.exists(full_filename_masks_3d):
                full_filename_masks = full_filename_masks_3d
                print_log("3D masks used !")
            else:
                full_filename_masks = full_filename_masks_2d
                print_log("2D masks used !")

            if os.path.exists(full_filename_masks):
                # loads and initializes masks
                segmented_masks = read_array(full_filename_masks)
                print_log(f"$ loaded mask file: {full_filename_masks}")

                # expands mask without overlap by a maximum of 'distance' pixels
                self.masks = expand_labels(
                    segmented_masks, distance=matrix_params.mask_expansion
                )

                # Generate new labels for each connected component to ensure each label is assigned to 1 mask only.
                self.masks = label(
                    self.masks > 0, connectivity=1
                )  # Relabel masks uniquely

                # initializes masks
                self.initializes_masks(self.masks)
                return True
            else:
                # Could not find a file with masks to assign. Report and continue with next ROI
                debug_mask_filename(
                    files_in_folder,
                    full_filename_masks,
                    self.mask_identifier.split("_")[0],
                    self.n_roi,
                    label=getattr(acq_params, f"{self.mask_identifier[:4]}_channel"),
                )
                return False

        else:
            print_log(
                f"$ Did not find any filename for mask: {self.mask_identifier}, channel: {channel}",
                "WARN",
            )
            print_dashes()
            # Could not find a file with masks to assign. Report and continue with next ROI
            debug_mask_filename(
                files_in_folder,
                "None",
                self.mask_identifier.split("_")[0],
                self.n_roi,
                label=getattr(acq_params, f"{self.mask_identifier[:4]}_channel"),
            )
            return False

    def assign_masks(
        self,
        output_filename,
        barcode_map,
        data_path,
        seg_params,
        acq_params: AcquisitionParams,
        matrix_params: MatrixParams,
    ):
        """
        Main function that:
            loads and processes barcode localization files, local alignment file, and masks
            initializes <cell_roi> class and assigns barcode localizations to masks
            then constructs the single cell PWD matrix and outputs it together with the contact map and the N-map.

        Parameters
        ----------
        output_filename : string
        self.current_param : Parameters Class
        self.current_folder : string

        self.pixel_size : dict, optional
            pixel_size = {'x': pixelSizeXY,
                        'y': pixelSizeXY,
                        'z': pixel_size_z}
            The default is 0.1 for x and y, 0.0 for z. Pixelsize in um

        self.log_name_md : str, optional
            Filename of Markdown output. The default is "log.md".
        self.ndims : int, optional
            indicates whether barcodes were localized in 2 or 3D. The default is 2.
        self.mask_identifier:

        Returns
        -------
        None.

        """

        # indexes localization tables by ROI
        barcode_map_roi = barcode_map.group_by("ROI #")
        number_rois = len(barcode_map_roi.groups.keys)

        print_dashes()
        print_log(
            f"> Loading masks and pre-processing barcodes for Mask <{self.mask_identifier}> for {number_rois} rois"
        )

        # finds TIFs in current_folder
        tif_files_in_folder = glob.glob(self.current_folder + os.sep + "*.tif")

        # loops over rois
        processing_order = 0

        for roi in range(number_rois):
            self.n_roi = barcode_map_roi.groups.keys[roi][
                0
            ]  # need to iterate over the first index
            self.barcode_map_roi = barcode_map.group_by("ROI #").groups[roi]

            if self.load_mask(
                tif_files_in_folder, data_path, seg_params, acq_params, matrix_params
            ):
                print_log(f"> Processing ROI# {self.n_roi}")

                # initializes trace table
                self.trace_table.initialize()

                # finds what barcodes are in each cell mask
                self.align_by_masking(matrix_params)
                print_log(
                    f"$ ROI: {roi}, N cells assigned: {self.n_cells_assigned - 1} out of {self.number_masks}\n"
                )

                # builds sc_distance_table
                self.builds_sc_distance_table()
                print_log(
                    f"$ Number of entries in trace table: {len(self.trace_table.data)}"
                )

                if len(self.trace_table.data) > 0:
                    # saves trace table with results per ROI
                    output_table_filename = f"{output_filename}_{self.label}_mask-{str(self.mask_identifier.split('_')[0])}_ROI-{str(self.n_roi)}.ecsv"

                    self.trace_table.save(output_table_filename, self.trace_table.data)

                    filepath_split = output_table_filename.split(".")[0].split(os.sep)
                    filepath_split.remove("data")
                    filepath_without_data_folder = (os.sep).join(filepath_split)
                    # plots results
                    self.trace_table.plots_traces(
                        [filepath_without_data_folder, "_traces_XYZ", ".png"],
                        masks=self.masks,
                    )

                    print_log(f"$ Saved output table as {output_table_filename}")
                else:
                    print_log("! Warning: table was empty therefore not saved!")

                processing_order += 1

    def build_trace_by_masking(
        self,
        barcode_map,
        data_path,
        seg_params,
        matrix_params,
        acq_params: AcquisitionParams,
    ):
        print_log(f"> Masks labels: {matrix_params.masks2process}")

        for mask_label in matrix_params.masks2process.keys():
            self.mask_identifier = matrix_params.masks2process[mask_label]
            self.mask_type = "DAPI" if "DAPI" in self.mask_identifier else "mask"
            tag = f"{str(self.ndims)}D"

            output_filename = (
                data_path
                + os.sep
                + matrix_params.folder
                + os.sep
                + "data"
                + os.sep
                + "Trace_"
                + tag
            )

            # creates and initializes trace table
            self.trace_table = ChromatinTraceTable()

            self.assign_masks(
                output_filename,
                barcode_map,
                data_path,
                seg_params,
                acq_params,
                matrix_params,
            )

            print_log(
                f"$ Trace built using mask assignment. Output saved in: {self.current_folder}",
                "info",
            )

    def group_localizations_by_coordinate(self, matrix_params: MatrixParams):
        """
        Uses a KDTree to group detections by it's coordinates, given a certain distance threshold
        Returns a list of lists. Each list contains the lines if the input data (segmentedObjects_3D_barcode.dat)
        where the detections are less than a pixel away from each other

        Parameters
        ----------
        coordinates : numpy array, float
            Matrix containing the xyz coordinates of barcodes.
        distance_threshold : float, default 1.0
            Distance threshold in pixels used to detect neighboring barcodes.

        Returns
        -------
        group_list : list
            list of lists containing the coordinates of barcodes associated together.
        """

        # gets coordinates from trace table
        data_table = self.barcode_map_roi
        pixel_size = self.pixel_size
        len_data_table = len(data_table)

        if self.ndims == 3:
            coordinates = np.concatenate(
                [
                    pixel_size["x"]
                    * data_table["xcentroid"].data.reshape(len_data_table, 1),
                    pixel_size["y"]
                    * data_table["ycentroid"].data.reshape(len_data_table, 1),
                    pixel_size["z"]
                    * data_table["zcentroid"].data.reshape(len_data_table, 1),
                ],
                axis=1,
            )
        elif self.ndims == 2:
            coordinates = np.concatenate(
                [
                    pixel_size["x"]
                    * data_table["xcentroid"].data.reshape(len_data_table, 1),
                    pixel_size["y"]
                    * data_table["ycentroid"].data.reshape(len_data_table, 1),
                    0.0 * data_table["zcentroid"].data.reshape(len_data_table, 1),
                ],
                axis=1,
            )

        # gets tree of coordinates
        print_log(f"> Creating KDTree for {self.ndims} dimensions")
        x_tree = KDTree(coordinates)

        # set distance threshold
        r = matrix_params.KDtree_distance_threshold_mum
        # Groups points when they're less than r away
        points = [x_tree.query_ball_point(element, r, p=2.0) for element in coordinates]

        # Get unique groups
        groups = list({tuple(x) for x in points})
        group_list = [list(x) for x in groups]
        self.n_cells_assigned = len(group_list)

        # Fills in trace information in localization table

        # iterates over traces
        for trace_id, trace in enumerate(group_list):
            # iterates over localizations in trace
            for i in range(len(trace)):
                # gets index of localization in data_table
                index_localization = trace[i]

                # records trace to which this index belongs
                data_table["CellID #"][index_localization] = trace_id

        self.barcode_map_roi = data_table

    def build_trace_by_clustering(
        self, barcode_map, data_path, matrix_params: MatrixParams
    ):
        # decompose by ROI!
        # indexes localization tables by ROI
        barcode_map_roi = barcode_map.group_by("ROI #")
        number_rois = len(barcode_map_roi.groups.keys)

        print_dashes()
        print_log(
            f"> Starting spatial clustering for {number_rois} ROI in {self.ndims} dimensions"
        )

        tag = f"{str(self.ndims)}D"

        output_filename = (
            data_path
            + os.sep
            + matrix_params.folder
            + os.sep
            + "data"
            + os.sep
            + "Trace_"
            + tag
        )

        # creates and initializes trace table
        self.trace_table = ChromatinTraceTable()

        # loops over rois
        for roi in range(number_rois):
            self.n_roi = barcode_map_roi.groups.keys[roi][
                0
            ]  # need to iterate over the first index
            self.barcode_map_roi = barcode_map.group_by("ROI #").groups[roi]

            print_log(f"$ Processing ROI# {self.n_roi}")

            # initializes trace table
            self.trace_table.initialize()

            # build traces by spatial clustering
            self.group_localizations_by_coordinate(matrix_params)
            print_log(f"$ ROI: {roi}, N cells assigned: {self.n_cells_assigned - 1}\n")

            # builds sc_distance_table
            self.builds_sc_distance_table()
            if len(self.trace_table.data) > 0:
                print_log(
                    f"$ Number of entries in trace table: {len(self.trace_table.data)}"
                )

                # saves trace table with results per ROI
                output_table_filename = (
                    output_filename
                    + "_"
                    + self.label
                    + "_KDtree"
                    + "_ROI-"
                    + str(self.n_roi)
                    + ".ecsv"
                )
                self.trace_table.save(output_table_filename, self.trace_table.data)

                filepath_split = output_table_filename.split(".")[0].split(os.sep)
                filepath_split.remove("data")
                filepath_without_data_folder = (os.sep).join(filepath_split)
                # plots results
                self.trace_table.plots_traces(
                    [filepath_without_data_folder, "_traces_XYZ", ".png"],
                    masks=self.masks,
                )

                print_log(
                    f"$ Traces built. Saved output table as {output_table_filename}"
                )
            else:
                print_log("! Warning: table was empty therefore not saved!")

    def launch_analysis(
        self,
        file,
        data_path,
        seg_params,
        matrix_params: MatrixParams,
        acq_params: AcquisitionParams,
    ):
        # loads barcode coordinate Tables
        table = LocalizationTable()
        barcode_map, self.unique_barcodes = table.load(file)
        print_log(f"$ {len(barcode_map)} localizations in {os.path.basename(file)}")
        if "3D" in os.path.basename(file):
            self.ndims = 3
            self.pixel_size = {
                "x": self.pixel_size_xy,
                "y": self.pixel_size_xy,
                "z": self.pixel_size_z,
            }
        else:
            self.ndims = 2
            self.pixel_size = {"x": self.pixel_size_xy, "y": self.pixel_size_xy, "z": 0}

        if "masking" in matrix_params.tracing_method:
            self.build_trace_by_masking(
                barcode_map, data_path, seg_params, matrix_params, acq_params
            )

        if (
            "clustering" in matrix_params.tracing_method and self.ndims == 3
        ):  # for now it only runs for 3D data
            self.build_trace_by_clustering(barcode_map, data_path, matrix_params)
        elif self.ndims == 2:
            print_log(
                "! Warning: localization files in 2D will not be processed using clustering.\n"
            )

    def run(self, data_path, seg_params, matrix_params, acq_params: AcquisitionParams):
        """
        Function that assigns barcode localizations to masks and constructs single cell cumulative PWD matrix.

        Parameters
        ----------
        current_param : class
            Parameters
        current_log : class
            logging class.

        Returns
        -------
        None.

        """
        self.label = "barcode"
        self.current_folder = data_path

        print_log(f"> Masks labels: {matrix_params.masks2process}")

        # iterates over barcode localization tables in the current folder
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
        files = list(glob.glob(data_file_base_2d + "_*" + self.label + ".dat"))
        files += list(glob.glob(data_file_base_3d + "_*" + self.label + ".dat"))
        # remove duplicate path, it's possible for example if 2d and 3d folder have same name
        files = list(set(files))
        if not files:
            print_log("$ No localization table found to process!", "WARN")
            return

        print_log(f"> Will process {len(files)} localization tables with names:")
        for file in files:
            print_log(f"{os.path.basename(file)}")

        for file in files:
            self.launch_analysis(file, data_path, seg_params, matrix_params, acq_params)

        print_log(f"$ {len(files)} barcode tables processed in {self.current_folder}")


def debug_mask_filename(
    files_in_folder, full_filename_masks, mask_identifier, n_roi, label=""
):
    print_log(f"# Error, no mask file found for ROI: {n_roi}\n")
    print_log(f"# File I was searching for: {full_filename_masks}")
    print_log("# Debug: ")
    for file in files_in_folder:
        if (
            file.split("_")[-1].split(".")[0] == label  # typically "ch00"
            and mask_identifier in file.split("_")
            and int(os.path.basename(file).split("_")[3]) == n_roi
        ):
            print_log("$ Hit found!")
        print_log(
            f'fileSplit:{file.split("_")[-1].split(".")[0]}, \
                {mask_identifier} in filename: {mask_identifier in os.path.basename(file).split("_")}, \
                ROI: {int(os.path.basename(file).split("_")[3])}'
        )


def binarize_coordinate(x):
    return np.nan if np.isnan(x) else int(x)
