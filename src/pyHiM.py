#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main file of pyHiM, include the top-level mechanism."""

__version__ = "0.10.0"

from datetime import datetime

import core.function_caller as fc
from core.data_manager import DataManager
from core.parameters import Parameters
from core.pyhim_logging import Logger, print_analyzing_label, print_log
from core.run_args import RunArgs


def main(command_line_arguments=None):
    """Main function of pyHiM

    Parameters
    ----------
    command_line_arguments : List[str], optional
        Used to give inputs for the runtime when you call this function like a module.
        For example, to test the pyHiM run from tests folder.
        By default None.
    """
    begin_time = datetime.now()

    run_args = RunArgs(command_line_arguments)

    logger = Logger(
        run_args.data_path,
        parallel=run_args.parallel,
        session_name="HiM_analysis",
        init_msg=run_args.args_to_str(),
    )

    datam = DataManager(
        run_args.data_path,
        logger.md_filename,
        param_file=run_args.params_path,
    )

    pipe = fc.Pipeline(datam, run_args.cmd_list, run_args.parallel, logger)
    pipe.lauch_dask_scheduler(threads_requested=run_args.thread_nbr, maximum_load=0.8)

    pipe.run()

    # No-refactored
    print_log("\n\n\n")
    raw_dict = datam.load_user_param()
    labels = datam.get_processable_labels()
    print_log(f"$ Labels to process: {labels}")
    for label in labels:
        # sets parameters with old way (temporary during pyHiM restructuration)
        current_param = Parameters(raw_dict, root_folder=datam.m_data_path, label=label)

        print_analyzing_label(f"Analyzing label: {label}")

        current_param.param_dict["parallel"] = pipe.parallel
        current_param.param_dict["fileNameMD"] = logger.md_filename

        # [applies registration to DAPI and barcodes]
        if "register_global" in pipe.cmds:
            projection_params = datam.labelled_params[label].projection
            registration_params = datam.labelled_params[label].registration
            pipe.apply_registrations(
                current_param,
                label,
                datam.m_data_path,
                registration_params,
                datam.processed_roi,
                projection_params,
            )

        # [aligns fiducials in 3D]
        if "register_local" in pipe.cmds:
            registration_params = datam.labelled_params[label].registration
            pipe.align_images_3d(
                current_param,
                label,
                datam.m_data_path,
                registration_params,
                datam.dict_shifts_path,
                datam.processed_roi,
                datam.acquisition_params.zBinning,
            )

        # [segments DAPI and sources in 2D]
        if ("mask_2d" in pipe.cmds and (label in ("DAPI", "mask"))) or (
            "localize_2d" in pipe.cmds and label == "barcode"
        ):
            segmentation_params = datam.labelled_params[label].segmentation
            pipe.segment_masks(
                current_param,
                label,
                datam.m_data_path,
                segmentation_params,
                datam.align_folder,
            )

        # [segments masks in 3D]
        if "mask_3d" in pipe.cmds and (label in ("DAPI", "mask")):
            registration_params = datam.labelled_params[label].registration
            segmentation_params = datam.labelled_params[label].segmentation
            pipe.segment_masks_3d(
                current_param,
                label,
                datam.processed_roi,
                datam.m_data_path,
                segmentation_params,
                datam.dict_shifts_path,
                datam.acquisition_params,
                registration_params,
            )
        # [align masks in 3D]
        if "shift_mask" in pipe.cmds and (label in ("DAPI", "mask")):
            registration_params = datam.labelled_params[label].registration
            segmentation_params = datam.labelled_params[label].segmentation
            pipe.shift_mask(
                current_param,
                label,
                datam.processed_roi,
                datam.m_data_path,
                segmentation_params,
                datam.dict_shifts_path,
                datam.acquisition_params,
                registration_params,
            )

        # [segments sources in 3D]
        if "localize_3d" in pipe.cmds and label == "barcode":
            projection_params = datam.labelled_params[label].projection
            registration_params = datam.labelled_params[label].registration
            segmentation_params = datam.labelled_params[label].segmentation
            pipe.segment_sources_3d(
                current_param,
                label,
                datam.processed_roi,
                datam.m_data_path,
                segmentation_params,
                datam.dict_shifts_path,
                datam.acquisition_params,
                projection_params,
                registration_params,
            )

        print_log("\n")
        del current_param

    for label in labels:
        # sets parameters with old way (temporary during pyHiM restructuration)
        current_param = Parameters(raw_dict, root_folder=datam.m_data_path, label=label)

        print_analyzing_label(f"Analyzing label: {label}")

        current_param.param_dict["parallel"] = pipe.parallel
        current_param.param_dict["fileNameMD"] = logger.md_filename

        # [filters barcode localization table]
        if "filter_localizations" in pipe.cmds and label == "barcode":
            segmentation_params = datam.labelled_params[label].segmentation
            matrix_params = datam.labelled_params[label].matrix
            fc.filter_localizations(
                current_param,
                label,
                datam.m_data_path,
                segmentation_params,
                matrix_params,
            )

        # [registers barcode localization table]
        if "register_localizations" in pipe.cmds and label == "barcode":
            registration_params = datam.labelled_params[label].registration
            segmentation_params = datam.labelled_params[label].segmentation
            matrix_params = datam.labelled_params[label].matrix
            fc.register_localizations(
                current_param,
                label,
                datam.m_data_path,
                datam.local_shifts_path,
                segmentation_params,
                registration_params,
                matrix_params,
            )

        # [build traces]
        if "build_traces" in pipe.cmds and label == "barcode":
            segmentation_params = datam.labelled_params[label].segmentation
            matrix_params = datam.labelled_params[label].matrix
            fc.build_traces(
                current_param,
                label,
                datam.m_data_path,
                segmentation_params,
                matrix_params,
                datam.acquisition_params,
            )

        # [builds matrices]
        if "build_matrix" in pipe.cmds and label == "barcode":
            matrix_params = datam.labelled_params[label].matrix
            fc.build_matrix(
                current_param,
                label,
                datam.m_data_path,
                matrix_params,
                datam.acquisition_params,
            )

        print_log("\n")
        del current_param

    # exits
    if pipe.parallel:
        pipe.m_dask.cluster.close()
        pipe.m_dask.client.close()

    del pipe

    print_log("\n==================== Normal termination ====================\n")
    print_log(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    print(f"[VERSION] pyHiM {__version__}")
    main()
