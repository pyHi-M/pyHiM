#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager module

Manage writing, reading and checking data.
"""

import os
import re
from dataclasses import asdict

from core.data_file import DataFile, NpyFile, TifFile, save_json
from core.parameters import AcquisitionParams, Params, deep_dict_update, load_json
from core.pyhim_logging import (
    print_log,
    print_section,
    print_session_name,
    print_title,
    write_string_to_file,
)


def extract_files(root: str):
    """Extract recursively file information of all files into a given directory.
    Note:
    * filepath is directory path with filename and extension
    * filename is the name without extension

    Parameters
    ----------
    root : str
        The name of root directory

    Returns
    -------
    List[Tuple(str,str,str)]
        List of file information: (filepath, filename, extension)
    """
    files = []
    # Iterate into dirname and each subdirectories dirpath, dirnames, filenames
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            split_filename = filename.split(".")
            extension = split_filename.pop() if len(split_filename) > 1 else None
            short_filename = ".".join(split_filename)
            filepath = os.path.join(dirpath, filename)
            files.append((filepath, short_filename, extension))

        if len(dirnames) > 0:
            print_log(f"$ Inside {dirpath}, subdirectories detected:\n  {dirnames}")

    return files


def remove_extension(filename: str):
    fl_split = filename.split(".")
    fl_split.pop()
    return ".".join(fl_split)


class DataManager:
    """Single party responsible for communicating data with the system"""

    def __init__(
        self,
        data_path: str,
        md_file: str = "",
        param_file: str = None,
    ):
        print_session_name("DataManager initialisation")
        self.m_data_path = self.__set_data_path(data_path)
        self.out_path = self.m_data_path
        self.md_log_file = md_file
        self.params_filename = "parameters"
        self.all_files = extract_files(self.m_data_path)
        self.param_file_path = self.find_param_file(param_file)
        self.tif_files = []
        self.ecsv_files = []
        self.npy_files = []
        self.filename_regex = ""
        self.label_decoder = self.__default_label_decoder()
        self.__processable_labels = {
            "fiducial": False,
            "DAPI": False,
            "mask": False,
            "barcode": False,
            "RNA": False,
        }
        self.processed_roi = None
        self.tif_ext = ["tif", "tiff"]
        self.npy_ext = ["npy"]
        self.ecsv_ext = ["ecsv", "table", "dat"]
        self.png_ext = ["png"]

        self.dict_shifts_path = ""
        self.local_shifts_path = ""
        self.align_folder = ""  # tempo refactoring attribute

        self.raw_dict = self.load_user_param_with_structure()
        print_section("acquisition")
        # pylint: disable=no-member
        self.acquisition_params = AcquisitionParams.from_dict(
            self.raw_dict["common"]["acquisition"]
        )
        self.labelled_params = {}

        self.set_up()

    @staticmethod
    def __set_data_path(data_path):
        return str(data_path) if data_path else os.getcwd()

    def find_param_file(self, param_file: str = None):
        """Find the user parameters file like `parameters.json` inside extracted input files.

        Returns
        -------
        str
            Parameters file path

        Raises
        ------
        ValueError
            Parameters file NOT FOUND
        """
        if param_file is not None:
            self.params_filename = param_file.split(os.sep)[-1].split(".")[0]
            if self.params_filename == "infoList":
                print_log(
                    "! 'infoList.json' is a DEPRECATED file name, please use by default 'parameters.json'.",
                    status="WARN",
                )
            return param_file
        for path, name, ext in self.all_files:
            if ext == "json" and name == self.params_filename:
                return str(path)
            # TODO: Remove this deprecated "infoList.json" including at pyHiM v1.0
            if ext == "json" and name == "infoList":
                print_log(
                    "! 'infoList.json' is a DEPRECATED file name, please use by default 'parameters.json'.",
                    status="WARN",
                )
                self.params_filename = "infoList"
                return str(path)
        # If we loop over all files, parameter file aren't detected.
        raise ValueError(
            f"Parameters file NOT FOUND, expected filename: {self.params_filename}.json OR infoList.json (DEPRECATED)"
        )

    @staticmethod
    def __default_label_decoder():
        return {
            "dapi_acq": {
                "ch00": "DAPI",
                "ch01": "fiducial",
                "ch02": "RNA",
            },
            "mask_acq": {
                "ch00": "mask",
                "ch01": "fiducial",
            },
            "barcode_acq": {
                "ch00": "barcode",
                "ch01": "fiducial",
            },
        }

    def create_dict_structure(self):
        return {
            "common": {
                "acquisition": {},
                "zProject": {},
                "alignImages": {},
                "segmentedObjects": {},
                "buildsPWDmatrix": {},
            },
            "labels": {
                "DAPI": {},
                "barcode": {},
                "fiducial": {},
                "RNA": {},
                "mask": {},
            },
        }

    def load_user_param_with_structure(self):
        dict_structure = self.create_dict_structure()
        return deep_dict_update(dict_structure, self.load_user_param())

    def create_out_structure(self, folder_name: str):
        """Create output folder structure for one Feature.

        Parameters
        ----------
        folder_name : str
            Relative path name of Feature output folder name
        """
        folder_path = self.out_path + os.sep + folder_name
        create_folder(folder_path)
        create_folder(folder_path + os.sep + "data")

    def add_to_processable_labels(self, label):
        self.__processable_labels[label] = True

    def get_processable_labels(self):
        label_list = []
        for label, value in self.__processable_labels.items():
            if value:
                label_list.append(label)
        return label_list

    def check_roi_uniqueness(self, roi_name: str):
        if self.processed_roi is None:
            print_log(f"$ Detected ROI: {roi_name}")
            self.processed_roi = roi_name
        elif self.processed_roi != roi_name:
            msg = f"""ERROR (ROI UNIQUENESS) - At least 2 ROI are detected: "{self.processed_roi}" and "{roi_name}"."""
            print_log(msg, status="DEBUG")
            raise SystemExit(msg)

    def dispatch_files(self):  # sourcery skip: remove-pass-elif
        """Get all input files and sort by extension type"""
        print_section("file names")
        unrecognized = 0
        for path, name, ext in self.all_files:
            if ext in self.tif_ext:
                parts = self.decode_file_parts(name)
                self.check_roi_uniqueness(parts["roi"])
                channel = parts["channel"][:4]
                label = self.find_label(name, channel)
                cycle = parts["cycle"]
                self.add_to_processable_labels(label)
                self.tif_files.append(TifFile(path, name, ext, label, cycle))
            elif ext in self.ecsv_ext:
                register_out_file = (
                    self.raw_dict.get("common", {})
                    .get("alignImages", {})
                    .get("outputFile")
                )
                register_local_alignment = (
                    self.raw_dict.get("common", {})
                    .get("alignImages", {})
                    .get("localAlignment")
                )
                name_to_find = (
                    str(register_out_file) + "_" + str(register_local_alignment)
                )
                if ext == "dat" and name == name_to_find:
                    self.local_shifts_path = path
                elif "barcode" in name:
                    self.add_to_processable_labels("barcode")
                self.ecsv_files.append((path, name, ext))
            elif ext in self.npy_ext:
                try:
                    parts = self.decode_file_parts(name)
                    self.check_roi_uniqueness(parts["roi"])
                    channel = parts["channel"][:4]
                    label = self.find_label(name, channel)
                    cycle = parts["cycle"]
                    self.add_to_processable_labels(label)
                    if "_2d_registered.npy" in path:  # tempo refactoring condition
                        self.align_folder = os.sep.join(path.split(os.sep)[:-1])
                        basename = name[: -len("_2d_registered")]
                        self.npy_files.append(
                            NpyFile(
                                None, "_2d_registered", cycle, path, basename, label
                            )
                        )
                    elif "_2d.npy" in path:
                        basename = name[:-3]
                        self.npy_files.append(
                            NpyFile(None, "_2d", cycle, path, basename, label)
                        )
                except ValueError:
                    unrecognized += 1
            elif ext == "json" and name == self.raw_dict.get("common", {}).get(
                "alignImages", {}
            ).get("outputFile"):
                self.dict_shifts_path = path
            elif ext in ["log", "md"] or (
                ext == "json" and name == self.params_filename
            ):
                pass
            else:
                unrecognized += 1
        print_log(f"! Unrecognized data files: {unrecognized}", status="WARN")

    def find_label(self, filename, channel):
        """Decode a filename to find its label (fiducial, DAPI, barcode, RNA, mask)

        Parameters
        ----------
        filename : str
            An input data filename

        Returns
        -------
        str
            A label (a type of data)

        Raises
        ------
        ValueError
            Label NOT FOUND
        """

        if "DAPI" in filename.split("_"):
            label = self.label_decoder["dapi_acq"][channel]
        elif "RT" in filename:
            label = self.label_decoder["barcode_acq"][channel]
        elif "mask" in filename:
            label = self.label_decoder["mask_acq"][channel]
        else:
            raise ValueError(f"Label NOT FOUND for this filename: {filename}")

        return label

    def set_label_decoder(self):
        self.label_decoder["dapi_acq"][self.acquisition_params.DAPI_channel] = "DAPI"
        self.label_decoder["dapi_acq"][
            self.acquisition_params.fiducialDAPI_channel
        ] = "fiducial"
        self.label_decoder["dapi_acq"][self.acquisition_params.RNA_channel] = "RNA"
        self.label_decoder["mask_acq"][self.acquisition_params.mask_channel] = "mask"
        self.label_decoder["mask_acq"][
            self.acquisition_params.fiducialMask_channel
        ] = "fiducial"
        self.label_decoder["barcode_acq"][
            self.acquisition_params.barcode_channel
        ] = "barcode"
        self.label_decoder["barcode_acq"][
            self.acquisition_params.fiducialBarcode_channel
        ] = "fiducial"

    def load_user_param(self):
        """Load user parameter JSON file like a Python dict

        Returns
        -------
        dict
            Python dict

        Raises
        ------
        ValueError
            file not found
        """
        params = load_json(self.param_file_path)
        if params is None:
            raise ValueError(f"Parameters file NOT FOUND: {self.param_file_path}")
        print_log(f"$ Parameters file read: {self.param_file_path}")
        return params

    def save_parameters_loaded(self):
        dict_struct = self.create_dict_structure()
        dict_struct["common"]["acquisition"] = asdict(self.acquisition_params)
        # remove "unknown_params" section
        dict_struct["common"]["acquisition"].pop("unknown_params", None)
        for label, params in self.labelled_params.items():
            label_dict = params.print_as_dict()
            for section, param_names in label_dict.items():  # for each section
                for key, val in param_names.items():  # for each parameter
                    rez = dict_struct["common"][section].get(key, None)
                    if rez is not None and rez != val:
                        dict_struct["labels"][label] = deep_dict_update(
                            dict_struct["labels"][label], {section: {key: val}}
                        )
                    else:
                        dict_struct["common"][section][key] = val
        save_json(dict_struct, os.path.join(self.m_data_path, "parameters_loaded.json"))

    def set_labelled_params(self, labelled_sections):
        print_session_name("Parameters initialisation")
        for label in self.get_processable_labels():
            print_title(f"Params: {label}")
            self.labelled_params[label] = Params(
                deep_dict_update(
                    self.raw_dict["common"], self.raw_dict["labels"][label]
                ),
                labelled_sections[label],
            )
        print_log("\n$ [Params] Initialisation done.\n")
        self.save_parameters_loaded()

    # TODO: clean this
    def set_up(self):
        # Regular expression
        self.filename_regex = remove_extension(self.acquisition_params.fileNameRegExp)
        # Channels
        self.set_label_decoder()
        self.dispatch_files()

    def decode_file_parts(self, file_name):
        """
        decodes variables from an input file. typically, RE takes the form:

        "scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\\w|-]+)" # pylint: disable=anomalous-backslash-in-string,line-too-long

        thus, by running decode_file_parts(file_name) you will get back
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
        if self.filename_regex:
            parts = re.search(self.filename_regex, file_name)
            if parts is None:
                raise ValueError(
                    f"Filename: {file_name}\nDoesn't match with regex: {self.filename_regex}"
                )
            if parts["runNumber"] is None:
                raise KeyError(f"'runNumber' part not found in this file:\n{file_name}")
            if parts["cycle"] is None:
                raise KeyError(f"'cycle' part not found in this file:\n{file_name}")
            if parts["roi"] is None:
                raise KeyError(f"'roi' part not found in this file:\n{file_name}")
            if parts["channel"] is None:
                raise KeyError(f"'channel' part not found in this file:\n{file_name}")
            return parts

        raise NameError("fileNameRegExp not found")

    def get_inputs(self, tif_labels: list[str], npy_labels: list[str]):
        if tif_labels:
            return [
                img_file for img_file in self.tif_files if img_file.label in tif_labels
            ]
        elif npy_labels:
            return [
                img_file for img_file in self.npy_files if img_file.label in npy_labels
            ]
        else:
            return []

    def save_data(self, results: list[DataFile], feature_folder: str, basename: str):
        files_to_keep = []
        for data_file in results:
            output_folder = self.m_data_path + os.sep + feature_folder
            data_file.save(output_folder, basename)
            data_file.delete_data()
            if data_file.extension in self.npy_ext:
                if "_2d_registered.npy" in data_file.path_name:
                    self.align_folder = os.sep.join(
                        data_file.path_name.split(os.sep)[:-1]
                    )
                parts = self.decode_file_parts(basename)
                self.check_roi_uniqueness(parts["roi"])
                channel = parts["channel"][:4]
                data_file.label = self.find_label(basename, channel)
                data_file.cycle = parts["cycle"]
                files_to_keep.append(data_file)
            elif data_file.extension in self.png_ext:
                write_string_to_file(
                    self.md_log_file,
                    f"{basename}\n ![]({data_file.path_name})\n",
                    "a",
                )
            elif data_file.extension == "json":
                self.dict_shifts_path = data_file.path_name
            # TODO: Add a self.local_shifts_path setter when register_local will be refactored
        return files_to_keep

    def __find_file_with_this_part(self, label_part, label, file_list):
        result = None
        for data_file in file_list:
            if data_file.label == label and label_part in data_file.basename.split("_"):
                result = data_file
        return result

    def load_reference(self, required_ref):
        if not required_ref:
            return None
        if required_ref["data_type"] == "npy":
            return self.__find_file_with_this_part(
                required_ref["label_part"], required_ref["label"], self.npy_files
            )
        else:
            raise ValueError


def create_folder(folder_path: str):
    """Create folder with `makedirs` from os module.
    It's a recursive directory creation function.

    Parameters
    ----------
    folder_path : str
        Path name of folder
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print_log(f"$ Folder '{folder_path}' created successfully.")
    else:
        print_log(f"! [INFO] Folder '{folder_path}' already exists.")
