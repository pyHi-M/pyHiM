#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main file of pyHiM, include the top-level mechanism."""

__version__ = "0.8.0"

import json
import os
import sys

# to remove in a future version
import warnings
from datetime import datetime

import apifish

import core.function_caller as fc
from core.data_manager import DataManager
from core.parameters import Parameters
from core.pyhim_logging import Logger, print_log
from core.run_args import RunArgs

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
        run_args.data_path, parallel=run_args.parallel, session_name="HiM_analysis"
    )

    datam = DataManager(
        run_args.data_path,
        logger,
        stardist_basename=run_args.stardist_basename,
        params_filename="infoList",
    )

    raw_dict = datam.load_user_param()
    global_param = Parameters(raw_dict, root_folder=datam.m_data_path)
    datam.set_up(global_param.get_sectioned_params("acquisition"))

    pipe = fc.Pipeline(
        datam,
        run_args.cmd_list,
        global_param,
        run_args.parallel,
        logger,
    )
    pipe.lauch_dask_scheduler(threads_requested=run_args.thread_nbr, maximum_load=0.8)

    labels = global_param.param_dict["labels"]
    print_log(f"$ Labels to process: {list(labels.keys())}")
    print_log(f"{json.dumps(labels, indent=4)}")

    pipe.run()

    for label in labels:
        # sets parameters
        current_param = Parameters(
            raw_dict,
            root_folder=datam.m_data_path,
            label=label,
            stardist_basename=datam.m_stardist_basename,
        )

        print_log("-----------------------------------------------------------------")
        print_log(
            f">                  Analyzing label: {current_param.param_dict['acquisition']['label']}           "
        )
        print_log("------------------------------------------------------------------")

        current_param.param_dict["parallel"] = pipe.parallel
        current_param.param_dict["fileNameMD"] = logger.md_filename

        # [projects 3D images in 2d]
        # If the projection cmd a require in parallel mode, we use the old way to run makeProjection
        # if pipe.parallel and "makeProjections" in run_args.cmd_list:
        #     pipe.make_projections(current_param)

        # [registers fiducials using a barcode as reference]
        if "alignImages" in run_args.cmd_list:
            pipe.align_images(current_param, label)

        # [applies registration to DAPI and barcodes]
        if "appliesRegistrations" in run_args.cmd_list:
            pipe.apply_registrations(current_param, label)

        # [aligns fiducials in 3D]
        if "alignImages3D" in run_args.cmd_list:
            pipe.align_images_3d(current_param, label)

        # [segments DAPI and sources in 2D]
        if "segmentMasks" in run_args.cmd_list:
            pipe.segment_masks(current_param, label)

        # [segments masks in 3D]
        if "segmentMasks3D" in run_args.cmd_list:
            pipe.segment_masks_3d(current_param, label)

        # [segments sources in 3D]
        if "segmentSources3D" in run_args.cmd_list:
            pipe.segment_sources_3d(current_param, label)

        # [filters barcode localization table]
        if "filter_localizations" in run_args.cmd_list:
            fc.filter_localizations(current_param, label)

        # [registers barcode localization table]
        if "register_localizations" in run_args.cmd_list:
            fc.register_localizations(current_param, label)

        # [build traces]
        if "build_traces" in run_args.cmd_list:
            fc.build_traces(current_param, label)

        # [builds matrices]
        if "build_matrix" in run_args.cmd_list:
            fc.build_matrix(current_param, label)

        # [builds PWD matrix for all folders with images]
        if "buildHiMmatrix" in run_args.cmd_list:
            pipe.process_pwd_matrices(current_param, label)

        print_log("\n")
        del current_param

    # exits
    logger.m_session.save()
    print_log("\n==================== Normal termination ====================\n")

    if run_args.parallel:
        pipe.m_dask.cluster.close()
        pipe.m_dask.client.close()
        # pipe.cluster.close()
        # pipe.client.close()

    del pipe

    print_log(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    if apifish.__version__ < "0.6.4dev":
        sys.exit("ERROR: Please update apifish (git checkout development && git pull)")
    else:
        main()
