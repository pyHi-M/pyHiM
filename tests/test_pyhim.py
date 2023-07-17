#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the non regression of modules called inside the jupyter notebook on pyHiM documentation"""

import os
import shutil
import tempfile

from core.data_manager import extract_files

# sys.path.append("..")
from pyHiM import main
from tests.testing_tools.comparison import (
    compare_ecsv_files,
    compare_line_by_line,
    compare_npy_files,
)

# build a temporary directory
tmp_dir = tempfile.TemporaryDirectory()
tmp_resources = os.path.join(tmp_dir.name, "resources")
shutil.copytree("pyhim-small-dataset/resources", tmp_resources)
tmp_small_inputs = os.path.join(tmp_resources, "small_dataset/IN")
tmp_stardist_basename = os.path.join(tmp_resources, "stardist_models")
tmp_traces_inputs = os.path.join(tmp_resources, "traces_dataset/IN")


def test_make_projections():
    """Check makeProjections"""
    main(["-F", tmp_small_inputs, "-C", "makeProjections", "-S", tmp_stardist_basename])
    tmp_z_project = os.path.join(tmp_small_inputs, "zProject")
    out_z_project = "pyhim-small-dataset/resources/small_dataset/OUT/makeProjections/"
    out_files = extract_files(out_z_project)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_z_project, filename)
        out_file = os.path.join(out_z_project, filename)
        assert compare_npy_files(tmp_file, out_file)


def test_align_images():
    """Check alignImages"""
    main(["-F", tmp_small_inputs, "-C", "alignImages", "-S", tmp_stardist_basename])
    tmp_align_images = os.path.join(tmp_small_inputs, "alignImages")
    out_align_images = "pyhim-small-dataset/resources/small_dataset/OUT/alignImages/"
    out_files = extract_files(out_align_images)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        if extension:
            filename = short_filename + "." + extension
        else:
            filename = short_filename
        tmp_file = os.path.join(tmp_align_images, filename)
        out_file = os.path.join(out_align_images, filename)
        if extension == "npy":
            assert compare_npy_files(tmp_file, out_file)
        else:
            assert compare_line_by_line(tmp_file, out_file, shuffled_lines=True)


def test_applies_registrations():
    """Check appliesRegistrations"""
    main(
        [
            "-F",
            tmp_small_inputs,
            "-C",
            "appliesRegistrations",
            "-S",
            tmp_stardist_basename,
        ]
    )
    tmp_align_images = os.path.join(tmp_small_inputs, "alignImages")
    out_align_images = (
        "pyhim-small-dataset/resources/small_dataset/OUT/appliesRegistrations/"
    )
    out_files = extract_files(out_align_images)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_align_images, filename)
        out_file = os.path.join(out_align_images, filename)
        assert compare_npy_files(tmp_file, out_file)


def test_align_images_3d():
    """Check alignImages3D"""
    main(["-F", tmp_small_inputs, "-C", "alignImages3D", "-S", tmp_stardist_basename])
    tmp_align_images = os.path.join(tmp_small_inputs, "alignImages")
    out_align_images = "pyhim-small-dataset/resources/small_dataset/OUT/alignImages3D/"
    out_files = extract_files(out_align_images)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_align_images, filename)
        out_file = os.path.join(out_align_images, filename)
        assert compare_line_by_line(tmp_file, out_file, shuffled_lines=True)


def test_segment_masks_3d():
    """Check segmentMasks3D"""
    main(["-F", tmp_small_inputs, "-C", "segmentMasks3D", "-S", tmp_stardist_basename])
    tmp_segmented_objects = os.path.join(tmp_small_inputs, "segmentedObjects")
    out_segmented_objects = (
        "pyhim-small-dataset/resources/small_dataset/OUT/segmentMasks3D/"
    )
    out_files = extract_files(out_segmented_objects)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_segmented_objects, filename)
        out_file = os.path.join(out_segmented_objects, filename)
        assert compare_npy_files(tmp_file, out_file)


# TODO: Find a way to test this module
# def test_segment_sources_3d():
#     """Check segmentSources3D"""
#     main(["-F", tmp_small_inputs, "-C", "segmentSources3D", "-S", tmp_stardist_basename])
#     tmp_segmented_objects = os.path.join(tmp_small_inputs, "segmentedObjects")
#     out_segmented_objects = "pyhim-small-dataset/resources/small_dataset/OUT/segmentSources3D/"
#     out_files = extract_files(out_segmented_objects)
#     assert len(out_files) > 0
#     for _, short_filename, extension in out_files:
#         filename = short_filename + "." + extension
#         tmp_file = os.path.join(tmp_segmented_objects, filename)
#         out_file = os.path.join(out_segmented_objects, filename)
#         assert compare_ecsv_files(tmp_file, out_file,columns_to_remove= ["Buid","id"], shuffled_lines=True)


def test_build_traces():
    """Check build_traces"""
    main(["-F", tmp_traces_inputs, "-C", "build_traces"])
    tmp_builds_pwd_matrix = os.path.join(tmp_traces_inputs, "buildsPWDmatrix")
    out_builds_pwd_matrix = (
        "pyhim-small-dataset/resources/traces_dataset/OUT/build_traces/"
    )
    out_files = extract_files(out_builds_pwd_matrix)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_builds_pwd_matrix, filename)
        out_file = os.path.join(out_builds_pwd_matrix, filename)
        assert compare_ecsv_files(tmp_file, out_file, columns_to_remove=["Trace_ID"])


def test_build_matrix():
    """Check build_matrix"""
    main(["-F", tmp_traces_inputs, "-C", "build_matrix"])
    tmp_builds_pwd_matrix = os.path.join(tmp_traces_inputs, "buildsPWDmatrix")
    out_builds_pwd_matrix = (
        "pyhim-small-dataset/resources/traces_dataset/OUT/build_matrix/"
    )
    out_files = extract_files(out_builds_pwd_matrix)
    assert len(out_files) > 0
    for _, short_filename, extension in out_files:
        filename = short_filename + "." + extension
        tmp_file = os.path.join(tmp_builds_pwd_matrix, filename)
        out_file = os.path.join(out_builds_pwd_matrix, filename)
        if extension == "npy":
            assert compare_npy_files(tmp_file, out_file, shuffled_plans=True)
        elif extension == "ecsv":
            assert compare_line_by_line(tmp_file, out_file)
