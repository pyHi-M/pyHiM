#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager module

Manage writing, reading and checking data.
"""

import json
import os


def extract_files(root: str):
    """Extract recursively file informations of all files into a given directory.
    Note: filename is the name without extension

    Parameters
    ----------
    root : str
        The name of root directory

    Returns
    -------
    List[Tuple(str,str,str)]
        List of file informations: (filepath, filename, extension)
    """
    files = []
    # Iterate into dirname and each subdirectories dirpath, dirnames, filenames
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            split_filename = filename.split(".")
            extension = split_filename.pop()
            short_filename = ".".join(split_filename)
            filepath = os.path.join(dirpath, filename)
            files.append((filepath, short_filename, extension))

        if len(dirnames) > 0:
            print("[INFO] Inside:")
            print(dirpath)
            print("[INFO] Subdirectories detected:")
            print(dirnames)

    return files


class DataManager:
    """Single party responsible for communicating data with the system"""

    def __init__(
        self,
        data_path: str,
        stardist_basename: str = "",
        filename_params: str = "infoList",
    ):
        self.m_data_path = self.__set_data_path(data_path)
        self.m_stardist_basename = str(stardist_basename)
        self.m_filename_params = filename_params
        self.all_files = extract_files(self.m_data_path)
        self.user_parameter = None
        self.data_images = []
        self.data_tables = []
        self.dispatch_files()

    @staticmethod
    def __set_data_path(data_path):
        if data_path:
            return str(data_path)
        return os.getcwd()

    def dispatch_files(self):
        """Get all input files and sort by extension type"""
        img_ext = ["tif", "tiff"]
        table_ext = ["csv", "ecsv", "dat"]
        for path, name, ext in self.all_files:
            if ext in img_ext:
                self.data_images.append((path, name, ext))
            elif ext in table_ext:
                self.data_tables.append((path, name, ext))
            elif ext == "json" and name == self.m_filename_params:
                self.user_parameter = str(path)
            else:
                print(
                    f"Unrecognized data file: {name}.{ext}\n>>>Inside this folder: {path}"
                )

    def load_user_param(self):
        """Load a json file but return None if file doesn't exist.

        Parameters
        ----------
        file_name : str
            JSON file name

        Returns
        -------
        dict
            Python dict
        """
        parameters = None

        file_name = self.user_parameter

        if os.path.exists(file_name):
            with open(file_name, encoding="utf-8") as json_file:
                parameters = json.load(json_file)
            print(f"$ Parameters file read: {file_name}")
        else:
            print(f"$ Parameters file NOT FOUND: {file_name}")

        return parameters
