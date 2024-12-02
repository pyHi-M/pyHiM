#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for file management
"""

import copy
import json
import os
import re
from dataclasses import asdict, dataclass, field
from os import path
from typing import Dict, List, Union

from dataclasses_json import CatchAll, LetterCase, Undefined, dataclass_json

from core.pyhim_logging import print_log, print_section, print_unknown_params


def load_json(file_name):
    """Load a JSON file like a python dict

    Parameters
    ----------
    file_name : str
        JSON file name

    Returns
    -------
    dict
        Python dict
    """
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as json_file:
            return json.load(json_file)
    return None


class Parameters:
    """
    Manage all pyHiM parameters.
    Old way, used before pyHiM restructuration.
    """

    def __init__(self, raw_dict, root_folder="./", label=""):
        self.files_to_process = []
        self.param_dict = self.complete_with_default(raw_dict)
        if label:
            self.param_dict = self.get_labelled_params(label)
        self.param_dict["rootFolder"] = root_folder
        self.file_parts = {}

    def complete_with_default(self, raw_dict):
        default = self.get_standard_parameters()
        return deep_dict_update(default, raw_dict)

    def get_sectioned_params(self, section_name: str):
        tempo = self.param_dict["common"].get(section_name)
        section_dict = {"common": tempo, "labels": {}}
        for key, value in self.param_dict["labels"].items():
            section_dict["labels"][key] = value.get(section_name)
        return section_dict

    def get_labelled_params(self, label_name: str):
        main_dict = self.param_dict["common"]
        main_dict["acquisition"]["label"] = label_name
        update = self.param_dict["labels"].get(label_name, {})
        return deep_dict_update(main_dict, update)

    def get_labeled_dict_value(self, section, param_name):
        labeled_dict = {}
        default = self.param_dict["common"][section][param_name]
        for key, value in self.param_dict["labels"].items():
            key_lower = key.lower()
            labeled_dict[key_lower] = value.get(section, {}).get(param_name, default)
        return labeled_dict

    def set_channel(self, key, default):
        """Set channel parameter with a default value

        Parameters
        ----------
        key : str
            Name like DAPI_channel, barcode_channel, fiducialMask_channel, ...
        default : str
            Like ch00, ch01, ...

        Returns
        -------
        str
            Channel value like 'ch02'
        """

        if key in self.param_dict["acquisition"].keys():
            return self.param_dict["acquisition"][key]
        return default

    def find_files_to_process(self, files_folder):
        """Find label-specific filenames from filename list.
        Save these filenames in self.files_to_process.

        Parameters
        ----------
        files_folder : list
            List of files
        """
        # defines channel for DAPI, fiducials and barcodes
        channel_dapi = self.set_channel("DAPI_channel", "ch00")
        channel_barcode = self.set_channel("barcode_channel", "ch01")
        channel_mask = self.set_channel("mask_channel", "ch01")
        channel_barcode_fiducial = self.set_channel("fiducialBarcode_channel", "ch00")
        channel_mask_fiducial = self.set_channel("fiducialMask_channel", "ch00")

        # finds if there are 2 or 3 channels for DAPI acquisition
        dapi_files = [
            file
            for file in files_folder
            if self.decode_file_parts(path.basename(file))["channel"] == "ch02"
            and "DAPI" in path.basename(file).split("_")
        ]

        # defines channels for RNA and DAPI-fiducial
        if dapi_files:
            channel_dapi_fiducial = self.set_channel("fiducialDAPI_channel", "ch02")
            channel_dapi_rna = self.set_channel("RNA_channel", "ch01")
        else:
            channel_dapi_fiducial = self.set_channel("fiducialDAPI_channel", "ch01")
            channel_dapi_rna = self.set_channel("RNA_channel", "ch04")

        if channel_dapi_fiducial and not dapi_files:
            print_log(
                "\n\n****You are using ch02 for channel_dapi_fiducial but there are only 2 channels for DAPI!\n\n",
                status="WARN",
            )

        # selects DAPI files
        if self.param_dict["acquisition"]["label"] == "DAPI":
            self.files_to_process = [
                file
                for file in files_folder
                if self.decode_file_parts(path.basename(file))["channel"]
                == channel_dapi
                and "DAPI" in path.basename(file).split("_")
            ]

        # selects DAPIch2 files
        elif self.param_dict["acquisition"]["label"] == "RNA":
            self.files_to_process = [
                file
                for file in files_folder
                if self.decode_file_parts(path.basename(file))["channel"]
                == channel_dapi_rna
                and "DAPI" in path.basename(file).split("_")
            ]

        # selects barcode files
        elif self.param_dict["acquisition"]["label"] == "barcode":
            self.files_to_process = [
                file
                for file in files_folder
                if len([i for i in file.split("_") if "RT" in i]) > 0
                and self.decode_file_parts(path.basename(file))["channel"]
                == channel_barcode
            ]

        # selects mask files
        elif self.param_dict["acquisition"]["label"] == "mask":
            self.files_to_process = [
                file
                for file in files_folder
                if len([i for i in file.split("_") if "mask" in i]) > 0
                and self.decode_file_parts(path.basename(file))["channel"]
                == channel_mask
            ]

        # selects fiducial files
        elif self.param_dict["acquisition"]["label"] == "fiducial":
            self.files_to_process = [
                file
                for file in files_folder
                if (
                    len([i for i in file.split("_") if "RT" in i]) > 0
                    and self.decode_file_parts(path.basename(file))["channel"]
                    == channel_barcode_fiducial
                )
                or (
                    len([i for i in file.split("_") if "mask" in i]) > 0
                    and self.decode_file_parts(path.basename(file))["channel"]
                    == channel_mask_fiducial
                )
                or (
                    "DAPI" in file.split("_")
                    and self.decode_file_parts(path.basename(file))["channel"]
                    == channel_dapi_fiducial
                )
            ]

        else:
            self.files_to_process = []

        print_log(f"$ Files to process: {len(self.files_to_process)}")
        for i, file in enumerate(self.files_to_process):
            print_log(f"{i}\t{os.path.basename(file)}")

    def decode_file_parts(self, file_name):
        # sourcery skip: use-named-expression
        """
        decodes variables from an input file. typically, RE takes the form:

        "scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\\w|-]+).tif" # pylint: disable=anomalous-backslash-in-string,line-too-long

        thus, by running decode_file_parts(current_param,file_name) you will get back
        either an empty dict if the RE were not present
        in your parameters.json file or a dict as follows if it all worked out fine:

        file_parts['runNumber']: runNumber number
        file_parts['cycle']: cycle string
        file_parts['roi']: roi number
        file_parts['channel']: channel string

        Parameters
        ----------
        file_name : string
            filename to decode

        Returns
        -------
        Dict with file_parts.

        """
        # decodes regular expressions
        regex = self.param_dict.get("acquisition").get("fileNameRegExp")
        return re.search(regex, file_name) if regex else None

    @staticmethod
    def get_standard_parameters():
        """Reference of the standard parameters"""
        return {
            "common": {
                "acquisition": {
                    "fileNameRegExp": "scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\\w|-]+).tif",
                    "DAPI_channel": "ch00",
                    "fiducialDAPI_channel": "ch01",
                    "RNA_channel": "ch02",
                    "fiducialBarcode_channel": "ch00",
                    "fiducialMask_channel": "ch00",
                    "barcode_channel": "ch01",
                    "mask_channel": "ch01",
                    "pixelSizeXY": 0.1,
                    "zBinning": 2,
                    "pixelSizeZ": 0.25,
                },  # barcode, fiducial
                "zProject": {
                    "folder": "project",  # output folder
                    "mode": "full",  # full, manual, automatic, laplacian
                    "blockSize": 256,
                    "display": True,
                    "zmin": 1,
                    "zmax": 59,
                    "zwindows": 15,
                    "windowSecurity": 2,
                    "zProjectOption": "MIP",  # sum or MIP
                },
                "alignImages": {
                    "register_global_folder": "register_global",
                    "register_local_folder": "register_local",
                    "outputFile": "shifts",
                    "referenceFiducial": "RT27",
                    "localAlignment": "block3D",  # options: None, mask2D, block3D
                    "alignByBlock": True,  # alignByBlock True will perform block alignment
                    # Used in blockAlignment to determine the % of error tolerated
                    "tolerance": 0.1,
                    # lower threshold to adjust image intensity levels
                    # before xcorrelation for alignment in 2D
                    "lower_threshold": 0.999,
                    # higher threshold to adjust image intensity levels
                    # before xcorrelation for alignment in 2D
                    "higher_threshold": 0.9999999,
                    # lower threshold to adjust image intensity levels
                    # before xcorrelation for Alignment3D
                    "3D_lower_threshold": 0.9,
                    # higher threshold to adjust image intensity levels
                    # before xcorrelation for Alignment3D
                    "3D_higher_threshold": 0.9999,
                    "background_sigma": 3.0,  # used to remove inhom background
                    "blockSize": 256,
                },
                "buildsPWDmatrix": {
                    "folder": "tracing",  # output folder
                    # available methods: masking, clustering
                    "tracing_method": ["masking", "clustering"],
                    # Expands masks until they collide by a max of 'mask_expansion' pixels
                    "mask_expansion": 8,
                    "flux_min": 10,  # min flux to keeep object
                    "flux_min_3D": 0.1,  # min flux to keeep object
                    "KDtree_distance_threshold_mum": 1,  # distance threshold used to build KDtree
                    # zxy tolerance used for block drift correction, in px
                    "toleranceDrift": [3, 1, 1],
                    # if True it will removed uncorrected localizations,
                    # otherwise they will remain uncorrectd.
                    "remove_uncorrected_localizations": True,
                },
                "segmentedObjects": {
                    "localize_2d_folder": "localize_2d",
                    "localize_3d_folder": "localize_3d",
                    "mask_2d_folder": "mask_2d",
                    "mask_3d_folder": "mask_3d",
                    "operation": "2D,3D",  # options: 2D or 3D
                    "outputFile": "segmentedObjects",
                    "background_method": "inhomogeneous",  # flat or inhomogeneous or stardist
                    "tessellation": True,  # tessellates masks
                    "background_sigma": 3.0,  # used to remove inhom background
                    "threshold_over_std": 1.0,  # threshold used to detect sources
                    "fwhm": 3.0,  # source size in px
                    "brightest": 1100,  # max number of sources segmented per FOV
                    "intensity_min": 0,  # min int to keep object
                    "intensity_max": 59,  # max int to keeep object
                    "area_min": 50,  # min area to keeep object
                    "area_max": 500,  # max area to keeep object
                    # options: 'thresholding' or 'stardist'
                    "3Dmethod": "stardist",
                    "residual_max": 2.5,  # z-profile Fit: max residuals to keeep object
                    "sigma_max": 5,  # z-profile Fit: max sigma 3D fitting to keeep object
                    # z-profile Fit: max diff between Moment and z-gaussian fits to keeep object
                    "centroidDifference_max": 5,
                    # z-profile Fit: window size to extract subVolume, px.
                    # 3 means subvolume will be 7x7.
                    "3DGaussianfitWindow": 3,
                    # constructs a YZ image by summing from xPlane-window:xPlane+window
                    "3dAP_window": 5,
                    "3dAP_flux_min": 2,  # # threshold to keep a source detected in YZ
                    "3dAP_brightest": 100,  # number of sources sought in each YZ plane
                    # px dist to attribute a source localized in YZ to one localized in XY
                    "3dAP_distTolerance": 1,
                    "3D_threshold_over_std": 5,
                    "3D_sigma": 3,
                    "3D_boxSize": 32,
                    "3D_area_min": 10,
                    "3D_area_max": 250,
                    "3D_nlevels": 64,
                    "3D_contrast": 0.001,
                    "3D_psf_z": 500,
                    "3D_psf_yx": 200,
                    "3D_lower_threshold": 0.99,
                    "3D_higher_threshold": 0.9999,
                    # if reducePlanes==True it will calculate focal plane and only use a region
                    # around it for segmentSources3D, otherwise will use the full stack
                    "reducePlanes": True,
                },
            },
            "labels": {
                "fiducial": {"order": 1},
                "barcode": {"order": 2},
                "DAPI": {"order": 3},
                "mask": {"order": 5},
                "RNA": {"order": 4},
            },
        }


def warn_default(key, val):
    if val != "None":
        print_log(
            f"""! key NOT FOUND inside parameters.json: "{key}"\n\t\t  Default value used: {val}""",
            status="WARN",
        )
    return val


def warn_pop(dico: dict, key: str, default):
    if dico.get(key):
        return dico.pop(key, default)
    return warn_default(key, default)


def set_default(key: str, val):
    # pylint: disable=invalid-field-call
    return field(default_factory=lambda: warn_default(key, val))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AcquisitionParams:
    """acquisition section of parameters.json parameter file."""

    # pylint: disable=invalid-name
    DAPI_channel: str = set_default("DAPI_channel", "None")
    RNA_channel: str = set_default("RNA_channel", "None")
    barcode_channel: str = set_default("barcode_channel", "None")
    mask_channel: str = set_default("mask_channel", "None")
    fiducialBarcode_channel: str = set_default("fiducialBarcode_channel", "None")
    fiducialMask_channel: str = set_default("fiducialMask_channel", "None")
    fiducialDAPI_channel: str = set_default("fiducialDAPI_channel", "None")
    fileNameRegExp: str = set_default(
        "fileNameRegExp",
        "scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\\w|-]+).tif",
    )
    pixelSizeXY: float = set_default("pixelSizeXY", 0.1)
    pixelSizeZ: float = set_default("pixelSizeZ", 0.25)
    zBinning: int = set_default("zBinning", 2)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print_unknown_params(self.unknown_params)


@dataclass_json(undefined=Undefined.INCLUDE, letter_case=LetterCase.CAMEL)
@dataclass
class ProjectionParams:
    """zProject section of parameters.json parameter file."""

    # pylint: disable=invalid-name
    folder: str = set_default("folder", "project")  # output folder
    mode: str = set_default("mode", "full")  # full, manual, automatic, laplacian
    block_size: int = set_default("block_size", 256)
    display: bool = set_default("display", True)
    zmin: int = set_default("zmin", 1)
    zmax: int = set_default("zmax", 59)
    zwindows: int = set_default("zwindows", 15)
    window_security: int = set_default("window_security", 2)
    z_project_option: str = set_default("z_project_option", "MIP")  # sum or MIP
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print_unknown_params(self.unknown_params)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class RegistrationParams:
    """alignImages section of parameters.json parameter file."""

    # pylint: disable=invalid-name
    register_global_folder: str = set_default(
        "register_global_folder", "register_global"
    )  # output folder
    register_local_folder: str = set_default(
        "register_local_folder", "register_local"
    )  # output folder
    outputFile: str = set_default("outputFile", "shifts")
    referenceFiducial: str = set_default("referenceFiducial", "RT27")
    localAlignment: str = set_default(
        "localAlignment", "block3D"
    )  # options: None, mask2D, block3D
    alignByBlock: bool = set_default(
        "alignByBlock", True
    )  # alignByBlock True will perform block alignment
    # Used in blockAlignment to determine the % of error tolerated
    tolerance: float = set_default("tolerance", 0.1)
    # lower threshold to adjust image intensity levels
    # before xcorrelation for alignment in 2D
    lower_threshold: float = set_default("lower_threshold", 0.999)
    # higher threshold to adjust image intensity levels
    # before xcorrelation for alignment in 2D
    higher_threshold: float = set_default("higher_threshold", 0.9999999)
    # lower threshold to adjust image intensity levels
    # before xcorrelation for Alignment3D
    _3D_lower_threshold: Union[float, str] = set_default("_3D_lower_threshold", "None")
    # higher threshold to adjust image intensity levels
    # before xcorrelation for Alignment3D
    _3D_higher_threshold: Union[float, str] = set_default(
        "_3D_higher_threshold", "None"
    )
    background_sigma: float = set_default(
        "background_sigma", 3.0
    )  # used to remove inhom background
    blockSize: int = set_default("blockSize", 256)  # register_global
    blockSizeXY: int = set_default("blockSizeXY", 128)  # register_local
    upsample_factor: int = set_default("upsample_factor", 100)  # register_local
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        self._3D_lower_threshold = (
            warn_pop(self.unknown_params, "3D_lower_threshold", 0.9)
            if self._3D_lower_threshold == "None"
            else self._3D_lower_threshold
        )
        self._3D_higher_threshold = (
            warn_pop(self.unknown_params, "3D_higher_threshold", 0.9999)
            if self._3D_higher_threshold == "None"
            else self._3D_higher_threshold
        )

        if self.unknown_params:  # if dict isn't empty
            print_unknown_params(self.unknown_params)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SegmentationParams:
    """segmentedObjects section of parameters.json parameter file."""

    # pylint: disable=invalid-name
    mask_2d_folder: str = set_default(
        "mask_2d_folder", "mask_2d"
    )  # output mask_2d folder
    mask_3d_folder: str = set_default("mask_3d_folder", "mask_3d")  # output folder
    localize_2d_folder: str = set_default(
        "localize_2d_folder", "localize_2d"
    )  # output folder
    localize_3d_folder: str = set_default(
        "localize_3d_folder", "localize_3d"
    )  # output folder
    operation: str = set_default("operation", "2D,3D")  # options: 2D or 3D
    outputFile: str = set_default("outputFile", "localizations")
    background_method: str = set_default(
        "background_method", "inhomogeneous"
    )  # flat or inhomogeneous or stardist
    stardist_basename: str = set_default("stardist_basename", "None")
    # network for 2D mask segmentation
    stardist_network: str = set_default("stardist_network", "None")
    # network for 3D mask or barcode segmentation
    stardist_network3D: str = set_default("stardist_network3D", "None")
    tessellation: bool = set_default("tessellation", True)  # tessellates masks
    background_sigma: float = set_default(
        "background_sigma", 3.0
    )  # used to remove inhom background
    threshold_over_std: float = set_default(
        "threshold_over_std", 1.0
    )  # threshold used to detect sources
    fwhm: float = set_default("fwhm", 3.0)  # source size in px
    brightest: int = set_default(
        "brightest", 1100
    )  # max number of sources segmented per FOV
    intensity_min: int = set_default("intensity_min", 0)  # min int to keep object
    intensity_max: int = set_default("intensity_max", 59)  # max int to keeep object
    area_min: int = set_default("area_min", 50)  # min area to keeep object
    area_max: int = set_default("area_max", 500)  # max area to keeep object
    # if reducePlanes==True it will calculate focal plane and only use a region
    # around it for segmentSources3D, otherwise will use the full stack
    reducePlanes: bool = set_default("reducePlanes", True)
    residual_max: float = set_default(
        "residual_max", 2.5
    )  # z-profile Fit: max residuals to keeep object
    sigma_max: int = set_default(
        "sigma_max", 5
    )  # z-profile Fit: max sigma 3D fitting to keeep object
    # z-profile Fit: max diff between Moment and z-gaussian fits to keeep object
    centroidDifference_max: int = set_default("centroidDifference_max", 5)
    # options: 'thresholding' or 'stardist'
    _3Dmethod: str = set_default("_3Dmethod", "None")
    # z-profile Fit: window size to extract subVolume, px.
    # 3 means subvolume will be 7x7.
    _3DGaussianfitWindow: Union[int, str] = set_default("_3DGaussianfitWindow", "None")
    # constructs a YZ image by summing from xPlane-window:xPlane+window
    _3dAP_window: Union[int, str] = set_default("_3dAP_window", "None")
    _3dAP_flux_min: Union[int, str] = set_default(
        "_3dAP_flux_min", "None"
    )  # # threshold to keep a source detected in YZ
    _3dAP_brightest: Union[int, str] = set_default(
        "_3dAP_brightest", "None"
    )  # number of sources sought in each YZ plane
    # px dist to attribute a source localized in YZ to one localized in XY
    _3dAP_distTolerance: Union[int, str] = set_default("_3dAP_distTolerance", "None")
    _3D_threshold_over_std: Union[int, str] = set_default(
        "_3D_threshold_over_std", "None"
    )
    _3D_sigma: Union[int, str] = set_default("_3D_sigma", "None")
    _3D_boxSize: Union[int, str] = set_default("_3D_boxSize", "None")
    _3D_area_min: Union[int, str] = set_default("_3D_area_min", "None")
    _3D_area_max: Union[int, str] = set_default("_3D_area_max", "None")
    _3D_nlevels: Union[int, str] = set_default("_3D_nlevels", "None")
    _3D_contrast: Union[float, str] = set_default("_3D_contrast", "None")
    _3D_psf_z: Union[int, str] = set_default("_3D_psf_z", "None")
    _3D_psf_yx: Union[int, str] = set_default("_3D_psf_yx", "None")
    _3D_lower_threshold: Union[float, str] = set_default("_3D_lower_threshold", "None")
    _3D_higher_threshold: Union[float, str] = set_default(
        "_3D_higher_threshold", "None"
    )
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        self._3Dmethod = (
            warn_pop(self.unknown_params, "3Dmethod", "stardist")
            if self._3Dmethod == "None"
            else self._3Dmethod
        )
        self._3DGaussianfitWindow = (
            warn_pop(self.unknown_params, "3DGaussianfitWindow", 3)
            if self._3DGaussianfitWindow == "None"
            else self._3DGaussianfitWindow
        )
        self._3dAP_window = (
            warn_pop(self.unknown_params, "3dAP_window", 5)
            if self._3dAP_window == "None"
            else self._3dAP_window
        )
        self._3dAP_flux_min = (
            warn_pop(self.unknown_params, "3dAP_flux_min", 2)
            if self._3dAP_flux_min == "None"
            else self._3dAP_flux_min
        )
        self._3dAP_brightest = (
            warn_pop(self.unknown_params, "3dAP_brightest", 100)
            if self._3dAP_brightest == "None"
            else self._3dAP_brightest
        )
        self._3dAP_distTolerance = (
            warn_pop(self.unknown_params, "3dAP_distTolerance", 1)
            if self._3dAP_distTolerance == "None"
            else self._3dAP_distTolerance
        )
        self._3D_threshold_over_std = (
            warn_pop(self.unknown_params, "3D_threshold_over_std", 5)
            if self._3D_threshold_over_std == "None"
            else self._3D_threshold_over_std
        )
        self._3D_sigma = (
            warn_pop(self.unknown_params, "3D_sigma", 3)
            if self._3D_sigma == "None"
            else self._3D_sigma
        )
        self._3D_boxSize = (
            warn_pop(self.unknown_params, "3D_boxSize", 32)
            if self._3D_boxSize == "None"
            else self._3D_boxSize
        )
        self._3D_area_min = (
            warn_pop(self.unknown_params, "3D_area_min", 10)
            if self._3D_area_min == "None"
            else self._3D_area_min
        )
        self._3D_area_max = (
            warn_pop(self.unknown_params, "3D_area_max", 250)
            if self._3D_area_max == "None"
            else self._3D_area_max
        )
        self._3D_nlevels = (
            warn_pop(self.unknown_params, "3D_nlevels", 64)
            if self._3D_nlevels == "None"
            else self._3D_nlevels
        )
        self._3D_contrast = (
            warn_pop(self.unknown_params, "3D_contrast", 0.001)
            if self._3D_contrast == "None"
            else self._3D_contrast
        )
        self._3D_psf_z = (
            warn_pop(self.unknown_params, "3D_psf_z", 500)
            if self._3D_psf_z == "None"
            else self._3D_psf_z
        )
        self._3D_psf_yx = (
            warn_pop(self.unknown_params, "3D_psf_yx", 200)
            if self._3D_psf_yx == "None"
            else self._3D_psf_yx
        )
        self._3D_lower_threshold = (
            warn_pop(self.unknown_params, "3D_lower_threshold", 0.99)
            if self._3D_lower_threshold == "None"
            else self._3D_lower_threshold
        )
        self._3D_higher_threshold = (
            warn_pop(self.unknown_params, "3D_higher_threshold", 0.9999)
            if self._3D_higher_threshold == "None"
            else self._3D_higher_threshold
        )
        if self.unknown_params:
            print_unknown_params(self.unknown_params)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class MatrixParams:
    """buildsPWDmatrix section of parameters.json parameter file."""

    # pylint: disable=invalid-name
    folder: str = set_default("folder", "tracing")  # output folder
    # available methods: masking, clustering
    tracing_method: List[str] = set_default("tracing_method", ["masking", "clustering"])
    # Expands masks until they collide by a max of 'mask_expansion' pixels
    mask_expansion: int = set_default("mask_expansion", 8)
    masks2process: Dict[str, str] = set_default(
        "masks2process", {"nuclei": "DAPI", "mask1": "mask0"}
    )
    flux_min: int = set_default("flux_min", 10)  # min flux to keeep object
    flux_min_3D: float = set_default("flux_min_3D", 0.1)  # min flux to keeep object
    KDtree_distance_threshold_mum: int = set_default(
        "KDtree_distance_threshold_mum", 1
    )  # distance threshold used to build KDtree
    # zxy tolerance used for block drift correction, in px
    toleranceDrift: Union[int, List[int]] = set_default("toleranceDrift", [3, 1, 1])
    # if True it will removed uncorrected localizations,
    # otherwise they will remain uncorrectd.
    remove_uncorrected_localizations: bool = set_default(
        "remove_uncorrected_localizations", True
    )
    z_offset: float = set_default("z_offset", 2.0)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print_unknown_params(self.unknown_params)


class Params:
    def __init__(self, labelled_dict: dict, sections: List[str]):
        self.acquisition = None
        self.projection = None
        self.registration = None
        self.segmentation = None
        self.matrix = None

        if "acquisition" in sections:
            print_section("acquisition")
            # pylint: disable=no-member
            self.acquisition = AcquisitionParams.from_dict(labelled_dict["acquisition"])
        if "projection" in sections:
            print_section("zProject")
            # pylint: disable=no-member
            self.projection = ProjectionParams.from_dict(labelled_dict["zProject"])
        if "registration" in sections:
            print_section("alignImages")
            # pylint: disable=no-member
            self.registration = RegistrationParams.from_dict(
                labelled_dict["alignImages"]
            )
        if "segmentation" in sections:
            print_section("segmentedObjects")
            # pylint: disable=no-member
            self.segmentation = SegmentationParams.from_dict(
                labelled_dict["segmentedObjects"]
            )
        if "matrix" in sections:
            print_section("buildsPWDmatrix")
            # pylint: disable=no-member
            self.matrix = MatrixParams.from_dict(labelled_dict["buildsPWDmatrix"])

        self.highlight_deprecated_params(labelled_dict)

    def highlight_deprecated_params(self, dict_to_check: dict):
        """Warns the user that there are unused/deprecated parameters in his parameters.json

        Parameters
        ----------
        dict_to_check : dict
            _description_
        """
        for key in dict_to_check:
            if key not in [
                "acquisition",
                "zProject",
                "alignImages",
                "segmentedObjects",
                "buildsPWDmatrix",
            ]:
                unused_params = {key: dict_to_check[key]}
                print_log(
                    f"! Unused parameters detected, it's probably a deprecated section: {unused_params}",
                    status="WARN",
                )

    def print_as_dict(self):
        result = {}
        attr_to_print = [
            (self.acquisition, "acquisition"),
            (self.projection, "zProject"),
            (self.registration, "alignImages"),
            (self.segmentation, "segmentedObjects"),
            (self.matrix, "buildsPWDmatrix"),
        ]
        for attribute, key in attr_to_print:
            if not (attribute is None or attribute == "None"):
                result[key] = asdict(attribute)
                # remove "unknown_params" section
                result[key].pop("unknown_params", None)
        return result


def load_alignment_dict(dict_shifts_path):
    """Load a JSON file with the shifts

    Parameters
    ----------
    dict_shifts_path : str
        Path to access at the shift dictionary, calculated by register_global

    Returns
    -------
    (dict,dict)
        Shift dictionaries
    """

    dict_shifts = load_json(dict_shifts_path)
    if dict_shifts is None:
        print_log(f"File with dictionary not found!: {dict_shifts_path}")
        dict_shifts_available = False
    else:
        print_log(f"Dictionary File loaded: {dict_shifts_path}")
        dict_shifts_available = True

    return dict_shifts, dict_shifts_available


# TODO: Can we replace this by json.dumps(dict,indent=4) ?
def print_dict(dictionary: dict):
    """Print parameters in your shell terminal

    Parameters
    ----------
    dictionary : dict
        Parameters dictionary
    """
    print_log("\n$ Parameters loaded:")
    for key in dictionary:
        spacer = "\t" * (3 - len(key) // 8)
        print_log(f"\t{key}{spacer}{dictionary[key]}")
    print_log("\n")


def get_dictionary_value(dictionary: dict, key: str, default: str = ""):
    """Get dict value with a default option if key doesn't exist.

    Parameters
    ----------
    dictionary : dict
        dictionary object
    key : str
        key for the dict
    default : str, optional
        default value is key doesn't exist, by default ""

    Returns
    -------
    str
        value or default
    """
    if key in dictionary:
        return dictionary[key]
    print_log(
        f"[WARNING] `{key}` not found, default value used: {default}", status="INFO"
    )
    return default


def loads_barcode_dict(file_name):
    """Loads a barcode type dict JSON

    Parameters
    ----------
    file_name : str
        JSON file name

    Returns
    -------
    dict
        Barcode dictionary
    """
    bc_dict = {}
    # Check if the file exists
    if not os.path.exists(file_name):
        print_log("File does not exist")
    else:
        # Opening JSON file
        with open(file_name, encoding="utf-8") as json_f:
            # returns JSON object as a dictionary
            barcode_type = json.load(json_f)
            print_log("$ {} barcode dictionary loaded")
            bc_dict = barcode_type

    return bc_dict


def deep_dict_update(main_dict: dict, new_dict: dict):
    """Update recursively a nested dict with another.
    main_dict keys/values that do not exist in new_dict are kept.

    Parameters
    ----------
    main_dict : dict
        Main dict with all default values
    new_dict : dict
        Dict with new values to update

    Returns
    -------
    dict
        The main_dict overwrite by new_dict value
    """
    main_deep_copy = copy.deepcopy(main_dict)
    for key, value in new_dict.items():
        if isinstance(value, dict):
            main_deep_copy[key] = deep_dict_update(main_deep_copy.get(key, {}), value)
        else:
            main_deep_copy[key] = value
    return main_deep_copy
