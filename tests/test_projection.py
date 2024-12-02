#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the non regression of projection feature,
- With the 4th modes: ["auto", "full", "manual", "laplacian"]
- And the both options: ["MIP","sum"]
  (inside the json file, DAPI is set for sum and others for MIP)
"""

import os
import shutil
import tempfile

from core.data_manager import extract_files

# sys.path.append("..")
from pyHiM import main
from tests.testing_tools.comparison import compare_npy_files, image_pixel_differences

# Build a temporary directory
tmp_dir = tempfile.TemporaryDirectory()
# Define a "projection" directory inside the temp dir
tmp_projection_in = os.path.join(tmp_dir.name, "projection")
# Copy the modes & IN/OUT structure for projection inside the "projection" temp dir
shutil.copytree("pyhim-small-dataset/projection/IN", tmp_projection_in)


def template_test_project(mode: str):
    """Check Project feature with all possibilities"""
    inputs = os.path.join(tmp_projection_in, mode)
    main(["-F", inputs, "-C", "project"])
    generated_z_project = os.path.join(inputs, "zProject")
    reference_outputs = f"pyhim-small-dataset/projection/OUT/{mode}/zProject/"
    generated_files = extract_files(generated_z_project)
    reference_files = extract_files(reference_outputs)
    assert len(generated_files) == len(reference_files)
    for filepath, short_filename, extension in generated_files:
        if "data" in filepath.split(os.sep):
            filename = f"data{os.sep}{short_filename}.{extension}"
        else:
            filename = f"{short_filename}.{extension}"
        tmp_file = os.path.join(generated_z_project, filename)
        out_file = os.path.join(reference_outputs, filename)
        assert os.path.exists(tmp_file)
        assert os.path.exists(out_file)
        if extension == "npy":
            assert compare_npy_files(tmp_file, out_file)
        elif extension == "png":
            assert image_pixel_differences(tmp_file, out_file)
        else:
            raise ValueError(f"Extension file UNRECOGNIZED: {filepath}")


def test_project_automatic():
    template_test_project("automatic")


def test_project_full():
    template_test_project("full")


def test_project_laplacian():
    template_test_project("laplacian")


def test_project_manual():
    template_test_project("manual")
