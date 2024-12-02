# -*- coding: utf-8 -*-
"""
Spyder Editor


Purpose: Corrects drift in 3D

The drift correction routines in alignImages.py take care of the corrections in XY but not in Z.

Drift in the z-position from cycle to cycle is small, typically 100-200 nm, but needs to be checked
and corrected. In addition, sample deformation can lead to inhomogeneous drift that cannot be corrected by a rigid body translation.

This routine solves both issues by correcting drift in 3D by block decomposition.

steps:
    - iterate over rois
    - load 3D fiducial file for reference fiducial
    - iterate over cycles <i>
    - load 3D fiducial file for fiducial barcode <i>
    - re-align 3D fiducial image using XY alignment
    - perform block alignment in 3D by cross=correlating blocks in 3D.

    - store in database.
    - display results:
        - drift correction maps in X-Y-Z
        - corrected blocks in XY, ZX, ZY

During buildMatrix this database is loaded, if available.
    - check database exist and load it
    - correct z-coordinate of the barcode provided the correction given in the dict

"""
# =============================================================================
# IMPORTS
# =============================================================================

import glob
import os
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np
from astropy.table import Table, vstack
from skimage import io
from skimage.registration import phase_cross_correlation

from core.dask_cluster import try_get_client
from core.parameters import RegistrationParams, load_alignment_dict, print_dict
from core.pyhim_logging import print_log, print_session_name
from core.saving import plot_3d_shift_matrices, plot_4_images
from imageProcessing.alignImages import (
    apply_xy_shift_3d_images,
    combine_blocks_image_by_reprojection,
    image_block_alignment_3d,
)
from imageProcessing.imageProcessing import preprocess_3d_image
from imageProcessing.makeProjections import reinterpolate_z

# =============================================================================
# CLASSES
# =============================================================================


class Drift3D:
    def __init__(self, param, registration_params, parallel=False):
        self.current_param = param
        self.reg_params = registration_params
        self.window = 3
        self.parallel = parallel
        self.p = {}
        self.image_ref_0 = None
        self.image_ref = None
        self.filenames_to_process_list = []
        self.filenames_with_ref_barcode = ""
        self.dict_shifts = None
        self.dict_shifts_available = None
        self.inner_parallel_loop = None

    def align_fiducials_3d_file(
        self,
        filename_to_process,
        data_path,
        params: RegistrationParams,
        roi_name,
        cycle_name,
        z_binning,
    ):
        """
        Aligns <filename_to_process> fiducial against reference

        Returns
        -------
        None.

        """

        p = self.p
        alignment_results_table = create_output_table()

        # excludes the reference fiducial and processes files in the same ROI
        inner_parallel_loop = self.inner_parallel_loop
        image_ref = self.image_ref
        image_ref_0 = self.image_ref_0
        dict_shifts_available = self.dict_shifts_available
        dict_shifts = self.dict_shifts
        output_folder = data_path + os.sep + params.register_local_folder

        return _align_fiducials_3d_file(
            filename_to_process,
            alignment_results_table,
            p,
            roi_name,
            cycle_name,
            inner_parallel_loop,
            image_ref,
            image_ref_0,
            dict_shifts,
            dict_shifts_available,
            output_folder,
            params,
            z_binning,
        )

    def load_reference_fiducial(
        self, filename_reference, z_binning, lower_threshold_3d, higher_threshold_3d
    ):
        """
        Loads Reference fiducial image

        Returns
        -------
        None.

        """

        self.p["fileNameReference"] = filename_reference
        print_log(f"Loading reference 3D image: {filename_reference}")

        self.image_ref_0, self.image_ref = load_n_preprocess_image(
            filename_reference,
            z_binning,
            lower_threshold_3d,
            higher_threshold_3d,
            parallel_execution=False,
        )

        self.image_ref_0 = np.sum(
            self.image_ref_0, axis=0
        )  # replaces 3D with a 2D projection

        print_log(f"$ Found {len(self.filenames_to_process_list)} files.")

    def load_dict_shifts(self, dict_shifts_path):
        """
        Lods dictionary of XY shifts

        Returns
        -------
        None.

        """
        print_log(f"""\nReference barcode: {self.reg_params.referenceFiducial}""")

        for file in self.current_param.files_to_process:
            if self.reg_params.referenceFiducial in file.split("_"):
                self.filenames_with_ref_barcode = file

        # loads dicShifts with shifts for all rois and all labels
        self.dict_shifts, self.dict_shifts_available = load_alignment_dict(
            dict_shifts_path
        )

    def align_fiducials_3d_in_folder(
        self,
        data_path,
        dict_shifts_path,
        params: RegistrationParams,
        roi_name,
        z_binning,
    ):
        """
        Refits all the barcode files found in root_folder

        Returns
        -------
        None.

        """
        now = datetime.now()
        print_dict(self.p)

        # gets files to process
        files_folder = glob.glob(data_path + os.sep + "*.tif")
        self.current_param.find_files_to_process(files_folder)

        # loads dictionary of shifts
        self.load_dict_shifts(dict_shifts_path)

        # creates Table that will hold results
        alignment_results_table_global = create_output_table()
        alignment_results_tables = []

        client = try_get_client()

        # loads reference fiducial image for this ROI
        self.load_reference_fiducial(
            self.filenames_with_ref_barcode,
            z_binning,
            params._3D_lower_threshold,
            params._3D_higher_threshold,
        )
        self.filenames_to_process_list = [
            x
            for x in self.current_param.files_to_process
            if (x != self.filenames_with_ref_barcode)
        ]
        number_files = len(self.filenames_to_process_list)

        if client is None:
            self.inner_parallel_loop = True
            for file_index, filename_to_process in enumerate(
                self.filenames_to_process_list
            ):
                print_log(f"\n\n>>>Iteration: {file_index+1}/{number_files}<<<")

                alignment_results_tables.append(
                    self.align_fiducials_3d_file(
                        filename_to_process,
                        data_path,
                        params,
                        roi_name,
                        find_cycle(self.current_param, filename_to_process),
                        z_binning,
                    )
                )

        else:
            self.inner_parallel_loop = False
            nb_workers = len(client.scheduler_info()["workers"])
            print_log(f"> Aligning {number_files} files using {nb_workers} workers...")

            futures = [
                client.submit(
                    self.align_fiducials_3d_file,
                    x,
                    data_path,
                    params,
                    roi_name,
                    find_cycle(self.current_param, x),
                    z_binning,
                )
                for x in self.filenames_to_process_list
            ]

            alignment_results_tables = client.gather(futures)
            print_log(
                f"> Retrieving {len(alignment_results_tables)} results from cluster"
            )

            # del futures

        # Merges Tables for different cycles and appends results Table to that of previous ROI
        alignment_results_table_global = vstack(
            [alignment_results_table_global] + alignment_results_tables
        )

        # saves Table with all shifts

        path_name = (
            data_path
            + os.sep
            + params.register_local_folder
            + os.sep
            + "data"
            + os.sep
            + params.outputFile
        )
        local_shifts_path = path_name + "_block3D.dat"

        alignment_results_table_global.write(
            local_shifts_path,
            format="ascii.ecsv",
            overwrite=True,
        )

        print_log(f"$ register_local processing time: {datetime.now() - now}")
        print_log(f"$ register_local output Table saved in: {local_shifts_path}")

        return local_shifts_path

    def align_fiducials_3d(
        self,
        data_path,
        params: RegistrationParams,
        dict_shifts_path,
        roi_name,
        z_binning,
    ):
        """
        runs refitting routine in root_folder

        Returns
        -------
        None.

        """
        session_name = "register_local"

        # processes folders and files
        print_session_name(session_name)

        print_log(f"-------> Processing Folder: {data_path}")
        # self.current_log.parallel = self.parallel

        local_shifts_path = self.align_fiducials_3d_in_folder(
            data_path, dict_shifts_path, params, roi_name, z_binning
        )

        print_log(f"HiM matrix in {data_path} processed")

        return local_shifts_path


# =============================================================================
#   FUNCTIONS
# =============================================================================


def find_cycle(param, filename):
    return str(param.decode_file_parts(os.path.basename(filename))["cycle"])


def load_n_preprocess_image(
    filename_to_process,
    z_binning,
    lower_threshold,
    higher_threshold,
    parallel_execution=True,
):
    print_log(f"$ File:{os.path.basename(filename_to_process)}")

    image_3d_0 = io.imread(filename_to_process).squeeze()

    # reinterpolates image in z if necessary
    image_3d_0 = reinterpolate_z(
        image_3d_0, range(0, image_3d_0.shape[0], z_binning), mode="remove"
    )

    image_3d = preprocess_3d_image(
        image_3d_0,
        lower_threshold,
        higher_threshold,
        parallel_execution=parallel_execution,
    )

    return image_3d_0, image_3d


def _align_fiducials_3d_file(
    filename_to_process,
    alignment_results_table,
    p,
    roi,
    cycle_name,
    inner_parallel_loop,
    image_ref,
    image_ref_0,
    dict_shifts,
    dict_shifts_available,
    output_folder,
    params: RegistrationParams,
    z_binning,
):
    # - load  and preprocesses 3D fiducial file
    print_log(f"\n\n>>>Processing roi:[{roi}] cycle:[{cycle_name}]<<<")
    image_3d_0, image_3d = load_n_preprocess_image(
        filename_to_process,
        z_binning,
        params._3D_lower_threshold,
        params._3D_higher_threshold,
        parallel_execution=inner_parallel_loop,
    )

    # shows original images and background subtracted
    image_3d_0 = np.sum(image_3d_0, axis=0)  # replaces by a 2D projection
    images = [image_ref, image_3d]
    images_2d = [np.sum(x, axis=0) for x in images]
    fig1 = plot_4_images(
        [image_ref_0, image_3d_0] + images_2d,
        titles=["reference", "cycle <i>", "processed reference", "processed cycle <i>"],
    )

    del image_3d_0

    # drifts 3D stack in XY
    # ---------------------
    if dict_shifts_available:
        # uses existing shift calculated by align_images
        try:
            shift = dict_shifts["ROI:" + roi][cycle_name]
            print_log("> Applying existing XY shift...")
        except KeyError:
            shift = None
            print_log(
                f"Could not find dictionary with alignment parameters for this ROI: ROI:{roi}, cycle: {cycle_name}",
                status="WARN",
            )
    if not dict_shifts_available or shift is None:
        # if dictionary of shift or key for this cycle was not found, then it will recalculate XY shift
        images_2d = [np.sum(x, axis=0) for x in images]

        print_log("> Calculating XY shift...")
        shift, _, _ = phase_cross_correlation(
            images_2d[0], images_2d[1], upsample_factor=params.upsample_factor
        )

    # applies XY shift to 3D stack
    # ----------------------------
    print_log(f"$ shifts XY = {shift}")

    # reinterpolate second file in XY using dictionary to get rough alignment
    images.append(
        apply_xy_shift_3d_images(
            image_3d, shift, parallel_execution=inner_parallel_loop
        )
    )

    del images[1], image_3d  # removes unshifted image to save memory

    # 3D image alignment by block
    # ---------------------------
    print_log("> Block-aligning images in 3D...")
    shift_matrices, block_ref, block_target = image_block_alignment_3d(
        images, block_size_xy=params.blockSizeXY, upsample_factor=params.upsample_factor
    )
    del images  # deletes image list to save memory

    # [plots shift matrices]
    fig2 = plot_3d_shift_matrices(shift_matrices, fontsize=8)

    # combines blocks into a single matrix for display instead of plotting a matrix of subplots each with a block
    outputs = []
    for axis in range(3):
        outputs.append(
            combine_blocks_image_by_reprojection(
                block_ref, block_target, shift_matrices=shift_matrices, axis1=axis
            )
        )

    mse_matrices = [x[2] for x in outputs]
    nrmse_matrices = [x[3] for x in outputs]

    fig3 = plt.figure(constrained_layout=False)
    fig3.set_size_inches((20 * 2, 20))
    grid_spec = fig3.add_gridspec(2, 2)
    ax = [
        fig3.add_subplot(grid_spec[:, 0]),
        fig3.add_subplot(grid_spec[0, 1]),
        fig3.add_subplot(grid_spec[1, 1]),
    ]

    titles = ["Z-projection", "X-projection", "Y-projection"]

    for axis, output, i in zip(ax, outputs, range(3)):
        axis.imshow(output[0])
        axis.set_title(titles[i])

    fig3.tight_layout()

    fig5 = plot_3d_shift_matrices(mse_matrices, fontsize=6, log=False, valfmt="{x:.2f}")
    fig5.suptitle("mean square root block matrices")

    # saves figures
    # -------------
    fig_titles = [
        "_bkgSubstracted.png",
        "_shiftMatrices.png",
        "_3Dalignments.png",
        "_MSEblocks.png",
    ]
    output_filenames = [
        output_folder + os.sep + os.path.basename(filename_to_process) + x
        for x in fig_titles
    ]

    figs = [fig1, fig2, fig3, fig5]
    for fig, file in zip(figs, output_filenames):
        fig.savefig(file)
        plt.close(fig)

    # Saves results
    # -------------
    # dict with shift_matrix and NRMSEmatrix: https://en.wikipedia.org/wiki/Root-mean-square_deviation
    # These matrices can be used to apply and assess zxy corrections for any pixel in the 3D image
    # reference file,aligned file,ROI,cycle,block_i,block_j,shift_z,shift_x,shift_y,quality_xy,quality_zy,quality_zx
    num_blocks, block_xy = block_ref.shape[0], block_ref.shape[-1]
    for i in range(num_blocks):
        for j in range(num_blocks):
            table_entry = [
                os.path.basename(p["fileNameReference"]),
                os.path.basename(filename_to_process),
                int(block_xy),
                int(roi),
                cycle_name,
                i,
                j,
                shift_matrices[0][i, j],
                shift_matrices[1][i, j],
                shift_matrices[2][i, j],
                nrmse_matrices[0][i, j],
                nrmse_matrices[1][i, j],
                nrmse_matrices[2][i, j],
            ]
            alignment_results_table.add_row(table_entry)

    # Erasing variables, TODO: check if it's necessary
    for var in dir():
        if var != "alignment_results_table":
            del var

    return alignment_results_table


def create_output_table():
    return Table(
        names=(
            "reference file",
            "aligned file",
            "blockXY",
            "ROI #",
            "label",
            "block_i",
            "block_j",
            "shift_z",
            "shift_x",
            "shift_y",
            "quality_xy",
            "quality_zy",
            "quality_zx",
        ),
        dtype=(
            "S2",
            "S2",
            "int",
            "int",
            "S2",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )
