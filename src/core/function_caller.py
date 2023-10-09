#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for high level function calling
"""


import os

from core.dask_cluster import DaskCluster
from core.pyhim_logging import print_log, print_session_name
from imageProcessing.alignImages import (
    align_images_in_current_folder,
    apply_registrations_to_current_folder,
    RegisterGlobal,
    ApplyRegisterGlobal,
)
from imageProcessing.alignImages3D import Drift3D
from imageProcessing.makeProjections import Project, Feature
from imageProcessing.segmentMasks import segment_masks
from imageProcessing.segmentMasks3D import Mask3D
from imageProcessing.segmentSources3D import Localize3D
from matrixOperations.alignBarcodesMasks import process_pwd_matrices
from matrixOperations.build_matrix import BuildMatrix
from matrixOperations.build_traces import BuildTraces
from matrixOperations.filter_localizations import FilterLocalizations
from matrixOperations.register_localizations import RegisterLocalizations


class Pipeline:
    """Class for high level function calling"""

    def __init__(self, data_m, cmd_list, is_parallel, logger):
        self.m_data_m = data_m
        self.cmds = self.interpret_cmd_list(cmd_list)
        self.set_params_from_cmds()
        self.parallel = is_parallel
        self.m_logger = logger
        self.m_dask = None
        self.features = []
        self.init_features()

    def interpret_cmd_list(self, cmd_list):
        cmds = []
        for cmd in cmd_list:
            if cmd.lower() in ["project", "makeprojection", "makeprojections"]:
                cmds.append("project")
            elif cmd.lower() in [
                "register_global",
                "registerglobal",
                "alignimage",
                "alignimages",
            ]:
                cmds.append("register_global")
            elif cmd.lower() in [
                "applyregistration",
                "applyregistrations",
                "appliesregistration",
                "appliesregistrations",
            ]:
                print_log(
                    f"! DEPRECATED COMMAND: {cmd}, now you can just use 'register_global'",
                    status="WARN",
                )
                cmds.append("register_global")
            if cmd.lower() in [
                "register_local",
                "registerlocal",
                "alignimages3d",
                "alignimage3d",
            ]:
                cmds.append("register_local")
            if cmd.lower() in [
                "mask_2d",
                "mask2d",
                "masks_2d",
                "masks2d",
                "segmentmasks",
                "segmentmask",
            ]:
                cmds.append("mask_2d")
            if cmd.lower() in [
                "localize_2d",
                "localize2d",
                "segmentmasks",
                "segmentmask",
            ]:
                cmds.append("localize_2d")
            if cmd.lower() in [
                "mask_3d",
                "mask3d",
                "masks_3d",
                "masks3d",
                "segmentmasks3d",
                "segmentmask3d",
            ]:
                cmds.append("mask_3d")
            if cmd.lower() in [
                "localize_3d",
                "localize3d",
                "segmentsource3d",
                "segmentsources3d",
            ]:
                cmds.append("localize_3d")
            if cmd.lower() in [
                "filter_localizations",
                "filter_localization",
                "filterlocalizations",
                "filterlocalization",
            ]:
                cmds.append("filter_localizations")
            if cmd.lower() in [
                "register_localizations",
                "register_localization",
                "registerlocalizations",
                "registerlocalization",
            ]:
                cmds.append("register_localizations")
            if cmd.lower() in [
                "build_traces",
                "build_trace",
                "buildtrace",
                "buildtraces",
            ]:
                cmds.append("build_traces")
            if cmd.lower() in [
                "build_matrix",
                "buildmatrix",
                "build_matrices",
                "buildmatrices",
            ]:
                cmds.append("build_matrix")
        # remove duplicate commands
        return list(set(cmds))

    def set_params_from_cmds(self):
        # TODO: precise association cmd<->section
        labelled_sections = {
            "barcode": [],
            "fiducial": [],
            "dapi": [],
            "rna": [],
            "mask": [],
        }

        if "project" in self.cmds:
            labelled_sections["barcode"].append("zProject")
            labelled_sections["fiducial"].append("zProject")
            labelled_sections["dapi"].append("zProject")
            labelled_sections["rna"].append("zProject")
            labelled_sections["mask"].append("zProject")

        if {
            "register_global",
            "register_local",
            "register_localizations",
        }.intersection(set(self.cmds)):
            labelled_sections["barcode"].append("alignImages")
            labelled_sections["fiducial"].append("alignImages")
            labelled_sections["dapi"].append("alignImages")
            labelled_sections["rna"].append("alignImages")
            labelled_sections["mask"].append("alignImages")

        if {"mask_2d", "mask_3d", "localize_2d", "localize_3d"}.intersection(
            set(self.cmds)
        ):
            labelled_sections["barcode"].append("segmentedObjects")
            labelled_sections["dapi"].append("segmentedObjects")
            labelled_sections["mask"].append("segmentedObjects")

        if {
            "filter_localizations",
            "register_localizations",
            "build_traces",
            "build_matrix",
        }.intersection(set(self.cmds)):
            labelled_sections["barcode"].append("buildsPWDmatrix")
            labelled_sections["dapi"].append("buildsPWDmatrix")
            labelled_sections["mask"].append("buildsPWDmatrix")

        self.m_data_m.set_labelled_params(labelled_sections)

    def _init_labelled_feature(
        self, feature_class_name: Feature, params_attr_name: str
    ):
        labelled_feature = {}
        for label in self.m_data_m.processable_labels:
            params_section = getattr(
                self.m_data_m.labelled_params[label], params_attr_name
            )
            labelled_feature[label] = feature_class_name(params_section)
        self.features.append(labelled_feature)

    def init_features(self):
        if "project" in self.cmds:
            self._init_labelled_feature(Project, "projection")
        # if "register_global" in self.cmds:
        #     self._init_labelled_feature(RegisterGlobal, "registration")
        #     self._init_labelled_feature(ApplyRegisterGlobal, "registration")

    def manage_parallel_option(self, feature, *args, **kwargs):
        if not self.parallel:
            feature(*args, **kwargs)
        else:
            result = self.m_dask.client.submit(feature, *args, **kwargs)
            _ = self.m_dask.client.gather(result)

    def lauch_dask_scheduler(self, threads_requested=25, maximum_load=0.8):
        if self.parallel:
            print_log(f"$ Requested {threads_requested} threads")
            self.m_dask = DaskCluster(threads_requested, maximum_load=maximum_load)
            # Run can be blocked with 0 or just 1 worker
            if self.m_dask.n_threads < 2:
                self.parallel = False
                print_log("! [WARNING] Resource too low to run in parallel mode.")
                print_log("! [WARNING] Sequential mode: activated")
            else:
                self.m_dask.create_distributed_client()

    def align_images(self, current_param, label, data_path, registration_params):
        if (
            label == "fiducial"
            and current_param.param_dict["acquisition"]["label"] == "fiducial"
        ):
            print_log(f"> Making image registrations for label: {label}")
            self.manage_parallel_option(
                align_images_in_current_folder,
                data_path,
                current_param,
                registration_params,
            )

    def align_images_3d(self, current_param, label):
        if (
            label == "fiducial"
            and current_param.param_dict["alignImages"]["localAlignment"] == "block3D"
        ):
            print_log(f"> Making 3D image registrations label: {label}")
            _drift_3d = Drift3D(
                current_param, self.m_logger.m_session, parallel=self.parallel
            )
            _drift_3d.align_fiducials_3d()

    def apply_registrations(self, current_param, label, data_path):
        if (
            label != "fiducial"
            and current_param.param_dict["acquisition"]["label"] != "fiducial"
        ):
            print_log(f"> Applying image registrations for label: {label}")
            self.manage_parallel_option(
                apply_registrations_to_current_folder, data_path, current_param
            )

    def segment_masks(self, current_param, label):
        if "segmentedObjects" in current_param.param_dict.keys():
            operation = current_param.param_dict["segmentedObjects"]["operation"]
        else:
            operation = [""]

        if (
            label != "RNA"
            and current_param.param_dict["acquisition"]["label"] != "RNA"
            and "2D" in operation
        ):
            self.manage_parallel_option(
                segment_masks, current_param, self.m_logger.m_session
            )

    def segment_masks_3d(self, current_param, label, roi_name: str):
        if (label in ("DAPI", "mask")) and "3D" in current_param.param_dict[
            "segmentedObjects"
        ]["operation"]:
            print_log(f"Making 3D image segmentations for label: {label}")
            print_log(f">>>>>>Label in functionCaller:{label}")

            _segment_sources_3d = Mask3D(
                current_param, self.m_logger.m_session, parallel=self.parallel
            )
            _segment_sources_3d.segment_masks_3d(roi_name)

    def segment_sources_3d(self, current_param, label, roi_name: str):
        if (
            label == "barcode"
            and "3D" in current_param.param_dict["segmentedObjects"]["operation"]
        ):
            print_log(f"Making 3D image segmentations for label: {label}")
            print_log(f">>>>>>Label in functionCaller:{label}")

            _segment_sources_3d = Localize3D(
                current_param, self.m_logger.m_session, roi_name, parallel=self.parallel
            )
            _segment_sources_3d.segment_sources_3d()

    def process_pwd_matrices(self, current_param, label):
        if label in ("DAPI", "mask"):
            self.manage_parallel_option(
                process_pwd_matrices, current_param, self.m_logger.m_session
            )

    def run(self):  # sourcery skip: remove-pass-body
        for feat_dict in self.features:
            feat = get_a_dict_value(feat_dict)
            (label_types, required_ref, required_table) = feat.get_required_inputs()
            # reference = self.m_data_m.load_reference(required_ref)
            # table = self.m_data_m.load_table(required_table)
            files_to_process = self.m_data_m.get_inputs(label_types)
            self.m_data_m.create_out_structure(feat.params.folder)
            if self.parallel:
                client = self.m_dask.client
                # forward_logging are used to allow workers send log msg to client with print_log()
                client.forward_logging()
                # Planify, for the future, work to execute in parallel
                threads = [
                    client.submit(run_pattern, feat_dict[f2p.label], f2p, self.m_data_m)
                    for f2p in files_to_process
                ]
                print_session_name("project")
                # Run workers
                client.gather(threads)

            else:
                print_session_name("project")
                for f2p in files_to_process:
                    run_pattern(feat_dict[f2p.label], f2p, self.m_data_m)


def run_pattern(feat, f2p, m_data_m):
    """Generic pattern for both run mode, sequential and parallel.
    (need to be a function and not a method for parallel running)

    Parameters
    ----------
    feat : Feature
        A sub-class of Feature
    f2p : ImageFile
        A file object with a `load` method.
        TODO: create a mother class `File` for ImageFile to be generic with other type of data files
    m_data_m : Allow to save outputs
    """
    data = f2p.load()
    print_log(f"\n> Analysing file: {os.path.basename(f2p.all_path)}")
    results = feat.run(data, f2p.label)
    # TODO: Include different type of inputs like reference image for registration or data table like ECSV
    # results = feat.run(data, reference, table)
    m_data_m.save_data(results, feat.find_out_tags(f2p.label), feat.params.folder, f2p)


# =============================================================================
# FUNCTIONS
# =============================================================================


def filter_localizations(current_param, label):
    """Filters barcode localization table

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        filter_localizations_instance = FilterLocalizations(current_param)
        filter_localizations_instance.filter_folder()


def register_localizations(current_param, label):
    """Registers barcode localization table

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        register_localizations_instance = RegisterLocalizations(current_param)
        register_localizations_instance.register()


def build_traces(current_param, label):
    """Build traces

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        build_traces_instance = BuildTraces(current_param)
        build_traces_instance.run()


def build_matrix(current_param, label):
    """Build matrices

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        build_matrix_instance = BuildMatrix(current_param)
        build_matrix_instance.run()


def get_a_dict_value(d: dict):
    return list(d.values())[0]
