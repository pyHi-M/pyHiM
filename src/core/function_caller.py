#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for high level function calling
"""


import os

from core.dask_cluster import DaskCluster
from core.parameters import (
    AcquisitionParams,
    MatrixParams,
    ProjectionParams,
    RegistrationParams,
    SegmentationParams,
)
from core.pyhim_logging import print_log, print_session_name
from imageProcessing import localize_3d, mask_3d
from imageProcessing.alignImages import (
    ApplyRegisterGlobal,
    RegisterGlobal,
    apply_registrations_to_current_folder,
)
from imageProcessing.alignImages3D import Drift3D
from imageProcessing.localize_2d import Localize2D
from imageProcessing.makeProjections import Feature, Project
from imageProcessing.mask_2d import Mask2D
from imageProcessing.register_local import RegisterLocal
from imageProcessing.segmentMasks import segment_masks
from imageProcessing.segmentMasks3D import Mask3D
from imageProcessing.segmentSources3D import Localize3D
from matrixOperations.build_matrix import BuildMatrix, BuildMatrixTempo
from matrixOperations.build_traces import BuildTraces, BuildTracesTempo
from matrixOperations.filter_localizations import (
    FilterLocalizations,
    FilterLocalizationsTempo,
)
from matrixOperations.register_localizations import (
    RegisterLocalizations,
    RegisterLocalizationsTempo,
)


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
            elif cmd.lower() in [
                "register_local",
                "registerlocal",
                "alignimages3d",
                "alignimage3d",
            ]:
                cmds.append("register_local")
            elif cmd.lower() in [
                "mask_2d",
                "mask2d",
                "masks_2d",
                "masks2d",
                "segmentmasks",
                "segmentmask",
            ]:
                cmds.append("mask_2d")
            elif cmd.lower() in [
                "localize_2d",
                "localize2d",
                "segmentmasks",
                "segmentmask",
            ]:
                cmds.append("localize_2d")
            elif cmd.lower() in [
                "mask_3d",
                "mask3d",
                "masks_3d",
                "masks3d",
                "segmentmasks3d",
                "segmentmask3d",
            ]:
                cmds.append("mask_3d")
            elif cmd.lower() in [
                "shift_mask",
                "shiftmask",
                "shift_masks",
                "shiftmasks",
            ]:
                cmds.append("shift_mask")
            elif cmd.lower() in [
                "localize_3d",
                "localize3d",
                "segmentsource3d",
                "segmentsources3d",
            ]:
                cmds.append("localize_3d")
            elif cmd.lower() in [
                "filter_localizations",
                "filter_localization",
                "filterlocalizations",
                "filterlocalization",
            ]:
                cmds.append("filter_localizations")
            elif cmd.lower() in [
                "register_localizations",
                "register_localization",
                "registerlocalizations",
                "registerlocalization",
            ]:
                cmds.append("register_localizations")
            elif cmd.lower() in [
                "build_traces",
                "build_trace",
                "buildtrace",
                "buildtraces",
            ]:
                cmds.append("build_traces")
            elif cmd.lower() in [
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
        self.labelled_sections = {
            "barcode": [],
            "fiducial": [],
            "DAPI": [],
            "RNA": [],
            "mask": [],
        }

        if {
            "project",
            "register_global",
            "mask_3d",
            "localize_3d",
        }.intersection(set(self.cmds)):
            self.labelled_sections["barcode"].append("projection")
            self.labelled_sections["fiducial"].append("projection")
            self.labelled_sections["DAPI"].append("projection")
            self.labelled_sections["RNA"].append("projection")
            self.labelled_sections["mask"].append("projection")

        if {
            "register_global",
            "register_local",
            "register_localizations",
            "mask_3d",
            "shift_mask",
            "localize_3d",
        }.intersection(set(self.cmds)):
            self.labelled_sections["barcode"].append("registration")
            self.labelled_sections["fiducial"].append("registration")
            self.labelled_sections["DAPI"].append("registration")
            self.labelled_sections["RNA"].append("registration")
            self.labelled_sections["mask"].append("registration")

        if {
            "mask_2d",
            "mask_3d",
            "shift_mask",
            "localize_2d",
            "localize_3d",
            "filter_localizations",
            "register_localizations",
            "build_traces",
        }.intersection(set(self.cmds)):
            self.labelled_sections["barcode"].append("segmentation")
            self.labelled_sections["DAPI"].append("segmentation")
            self.labelled_sections["mask"].append("segmentation")

        if {
            "filter_localizations",
            "register_localizations",
            "build_traces",
            "build_matrix",
        }.intersection(set(self.cmds)):
            self.labelled_sections["barcode"].append("matrix")
            self.labelled_sections["DAPI"].append("matrix")
            self.labelled_sections["mask"].append("matrix")

        self.m_data_m.set_labelled_params(self.labelled_sections)

    def _init_labelled_feature(
        self, feature_class_name: Feature, params_attr_name: str
    ):
        labelled_feature = {}
        for label in self.m_data_m.get_processable_labels():
            if params_attr_name in self.labelled_sections[label]:
                params_section = getattr(
                    self.m_data_m.labelled_params[label], params_attr_name
                )
                labelled_feature[label] = feature_class_name(params_section)
        self.features.append(labelled_feature)

    def init_features(self):
        ordered_routines = []
        if "project" in self.cmds:
            self._init_labelled_feature(Project, "projection")
            ordered_routines.append("project")
        if "register_global" in self.cmds:
            self._init_labelled_feature(RegisterGlobal, "registration")
            self._init_labelled_feature(ApplyRegisterGlobal, "registration")
            ordered_routines.append("register_global")
        if "register_local" in self.cmds:
            self._init_labelled_feature(RegisterLocal, "registration")
            ordered_routines.append("register_local")
        if "mask_2d" in self.cmds:
            self._init_labelled_feature(Mask2D, "segmentation")
            ordered_routines.append("mask_2d")
        if "localize_2d" in self.cmds:
            self._init_labelled_feature(Localize2D, "segmentation")
            ordered_routines.append("localize_2d")
        if "mask_3d" in self.cmds:
            self._init_labelled_feature(mask_3d.Mask3D, "segmentation")
            ordered_routines.append("mask_3d")
        if "localize_3d" in self.cmds:
            self._init_labelled_feature(localize_3d.Localize3D, "segmentation")
            ordered_routines.append("localize_3d")
        if "filter_localizations" in self.cmds:
            self._init_labelled_feature(FilterLocalizationsTempo, "matrix")
            ordered_routines.append("filter_localizations")
        if "register_localizations" in self.cmds:
            self._init_labelled_feature(RegisterLocalizationsTempo, "matrix")
            ordered_routines.append("register_localizations")
        if "build_traces" in self.cmds:
            self._init_labelled_feature(BuildTracesTempo, "matrix")
            ordered_routines.append("build_traces")
        if "build_matrix" in self.cmds:
            self._init_labelled_feature(BuildMatrixTempo, "matrix")
            ordered_routines.append("build_matrix")

        print_log(f"$ Ordered list of routines to run:\n{ordered_routines}")

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

    def align_images_3d(
        self,
        current_param,
        label,
        data_path,
        registration_params: RegistrationParams,
        dict_shifts_path,
        roi_name,
        z_binning,
    ):
        if label == "fiducial" and registration_params.localAlignment == "block3D":
            print_log(f"> Making 3D image registrations label: {label}")
            _drift_3d = Drift3D(
                current_param, registration_params, parallel=self.parallel
            )
            local_shifts_path = _drift_3d.align_fiducials_3d(
                data_path, registration_params, dict_shifts_path, roi_name, z_binning
            )
            self.m_data_m.local_shifts_path = local_shifts_path

    def apply_registrations(
        self,
        current_param,
        label,
        data_path,
        registration_params,
        roi_name,
        projection_params,
    ):
        if label != "fiducial":
            print_log(f"> Applying image registrations for label: {label}")
            self.manage_parallel_option(
                apply_registrations_to_current_folder,
                data_path,
                current_param,
                registration_params,
                roi_name,
                projection_params,
            )

    def segment_masks(
        self, current_param, label, data_path, params: SegmentationParams, align_folder
    ):
        if "segmentedObjects" in current_param.param_dict.keys():
            operation = current_param.param_dict["segmentedObjects"]["operation"]
        else:
            operation = [""]

        if label != "RNA" and "2D" in operation:
            self.manage_parallel_option(
                segment_masks, current_param, data_path, params, align_folder, label
            )

    def segment_masks_3d(
        self,
        current_param,
        label,
        roi_name: str,
        data_path,
        segmentation_params,
        dict_shifts_path,
        acq_params: AcquisitionParams,
        reg_params: RegistrationParams,
    ):
        if (label in ("DAPI", "mask")) and "3D" in current_param.param_dict[
            "segmentedObjects"
        ]["operation"]:
            print_log(f"Making 3D image segmentations for label: {label}")
            print_log(f">>>>>>Label in functionCaller:{label}")

            _segment_sources_3d = Mask3D(
                current_param,
                parallel=self.parallel,
            )
            _segment_sources_3d.segment_masks_3d(
                roi_name,
                data_path,
                dict_shifts_path,
                segmentation_params,
                acq_params,
                reg_params.referenceFiducial,
            )

    def shift_mask(
        self,
        current_param,
        label,
        roi_name: str,
        data_path,
        segmentation_params,
        dict_shifts_path,
        acq_params: AcquisitionParams,
        reg_params: RegistrationParams,
    ):
        if (label in ("DAPI", "mask")) and "3D" in current_param.param_dict[
            "segmentedObjects"
        ]["operation"]:
            print_log(f"Making 3D mask shift for label: {label}")
            _mask_3d = Mask3D(
                current_param,
                parallel=self.parallel,
            )
            _mask_3d.shift_mask(
                roi_name,
                data_path,
                dict_shifts_path,
                segmentation_params,
                acq_params,
                reg_params.referenceFiducial,
                label,
            )

    def segment_sources_3d(
        self,
        current_param,
        label,
        roi_name: str,
        data_path,
        segmentation_params,
        dict_shifts_path,
        acq_params: AcquisitionParams,
        proj_params: ProjectionParams,
        reg_params: RegistrationParams,
    ):
        if (
            label == "barcode"
            and "3D" in current_param.param_dict["segmentedObjects"]["operation"]
        ):
            print_log(f"Making 3D image segmentations for label: {label}")
            print_log(f">>>>>>Label in functionCaller:{label}")

            _segment_sources_3d = Localize3D(
                current_param,
                roi_name,
                acq_params,
                proj_params,
                reg_params,
                segmentation_params,
                parallel=self.parallel,
            )
            _segment_sources_3d.segment_sources_3d(
                data_path, dict_shifts_path, segmentation_params
            )

    def run(self):  # sourcery skip: remove-pass-body
        for feat_dict in self.features:
            feat = get_a_dict_value(feat_dict)
            (
                tif_labels,
                npy_labels,
                required_ref,
                required_table,
            ) = feat.get_required_inputs()
            reference_file = self.m_data_m.load_reference(required_ref)
            # table = self.m_data_m.load_table(required_table)

            files_to_process = self.m_data_m.get_inputs(tif_labels, npy_labels)

            self.m_data_m.create_out_structure(feat.out_folder)
            results_to_keep = []
            files_to_keep = []
            if self.parallel:
                client = self.m_dask.client
                # forward_logging are used to allow workers send log msg to client with print_log()
                client.forward_logging()
                # Planify, for the future, work to execute in parallel
                threads = [
                    client.submit(
                        run_pattern,
                        feat_dict[f2p.label],
                        f2p,
                        reference_file,
                        self.m_data_m,
                    )
                    for f2p in files_to_process
                ]
                print_session_name(feat.name)
                # Run workers
                collect = client.gather(threads)
                for results, npy_files in collect:
                    results_to_keep.append(results)
                    files_to_keep += npy_files

            else:
                print_session_name(feat.name)
                for f2p in files_to_process:
                    results, npy_files = run_pattern(
                        feat_dict[f2p.label], f2p, reference_file, self.m_data_m
                    )
                    results_to_keep.append(results)
                    files_to_keep += npy_files

            merged_results = feat.merge_results(remove_none_from_list(results_to_keep))
            out_filename = getattr(feat.params, "outputFile", "")
            npy_files = self.m_data_m.save_data(
                merged_results, feat.out_folder, out_filename
            )
            self.m_data_m.npy_files += files_to_keep + npy_files


def run_pattern(feat, f2p, reference_file, m_data_m):
    """Generic pattern for both run mode, sequential and parallel.
    (need to be a function and not a method for parallel running)

    Parameters
    ----------
    feat : Feature
        A sub-class of Feature
    f2p : TifFile
        A file object with a `load` method.
        TODO: create a mother class `File` for TifFile to be generic with other type of data files
    m_data_m : Allow to save outputs
    """
    data = f2p.load()

    reference = reference_file.load() if reference_file else None
    print_log(f"\n> Analysing file: {os.path.basename(f2p.path_name)}")
    results_to_save, results_to_keep = feat.run(data, reference)
    # TODO: Include different type of inputs like reference image for registration or data table like ECSV
    # results = feat.run(data, reference, table)
    files_to_keep = m_data_m.save_data(results_to_save, feat.out_folder, f2p.basename)
    if results_to_keep is not None:
        results_to_keep["tif_name"] = f2p.tif_name
        results_to_keep["cycle"] = f2p.cycle
        results_to_keep["roi"] = m_data_m.processed_roi
        results_to_keep["ref_tif_name"] = (
            reference_file.tif_name if reference_file else None
        )
    return results_to_keep, files_to_keep


def remove_none_from_list(list_with_none: list):
    return [x for x in list_with_none if x is not None]


# =============================================================================
# FUNCTIONS
# =============================================================================


def filter_localizations(
    current_param,
    label,
    data_path,
    segmentation_params,
    matrix_params: MatrixParams,
):
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
        filter_localizations_instance.filter_folder(
            data_path, segmentation_params, matrix_params
        )


def register_localizations(
    current_param,
    label,
    data_path,
    local_shifts_path,
    segmentation_params,
    reg_params: RegistrationParams,
    matrix_params: MatrixParams,
):
    """Registers barcode localization table

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        register_localizations_instance = RegisterLocalizations(
            current_param, matrix_params
        )
        register_localizations_instance.register(
            data_path, local_shifts_path, segmentation_params, reg_params
        )


def build_traces(
    current_param,
    label,
    data_path,
    segmentation_params,
    matrix_params: MatrixParams,
    acq_params: AcquisitionParams,
):
    """Build traces

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        build_traces_instance = BuildTraces(current_param, acq_params)
        build_traces_instance.run(
            data_path, segmentation_params, matrix_params, acq_params
        )


def build_matrix(
    current_param, label, data_path, matrix_params, acq_params: AcquisitionParams
):
    """Build matrices

    Parameters
    ----------
    current_param : Parameters
        _description_
    label : str
        Only 'barcode' are accepted
    """
    if label == "barcode":
        build_matrix_instance = BuildMatrix(current_param, acq_params)
        build_matrix_instance.run(data_path, matrix_params)


def get_a_dict_value(d: dict):
    return list(d.values())[0]
