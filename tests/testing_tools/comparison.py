#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for pyHiM testing
"""


import numpy as np
from astropy.table import Table
from PIL import Image  # 'Image' is used to load images
from PIL import ImageChops  # 'ImageChops' for arithmetical image operations


def image_pixel_differences(base_path, compare_path):
    """
    Calculates the bounding box of the non-zero regions in the image.
    :param base_image: target image to find
    :param compare_image:  image containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    base_image = Image.open(base_path)
    compare_image = Image.open(compare_path)
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.
    diff = ImageChops.difference(base_image, compare_image)
    return not bool(diff.getbbox())


def compare_npy_files(first_file, second_file, shuffled_plans=False):
    """Load both files as numpy array and compare them.

    Args:
        first_file (string): first complete file path
        second_file (string): second complete file path

    Returns:
        bool: True if they are the same array with the same value
    """
    first_npy = np.load(first_file)
    second_npy = np.load(second_file)
    is_same = np.array_equal(first_npy, second_npy, equal_nan=True)
    if not is_same and shuffled_plans:
        reverse_first = np.swapaxes(first_npy, 0, 2)
        reverse_second = np.swapaxes(second_npy, 0, 2)
        is_same = True
        for plan in reverse_first:
            is_inside = (
                (np.equal(plan, reverse_second) | np.isnan(reverse_second))
                .all((1, 2))
                .any()
            )
            is_same = is_same and is_inside
    return is_same


def compare_ecsv_files(
    first_file, second_file, columns_to_remove: list[str] = None, shuffled_lines=False
):
    first_ecsv = Table.read(first_file, format="ascii.ecsv")
    second_ecsv = Table.read(second_file, format="ascii.ecsv")

    if columns_to_remove:
        for col in columns_to_remove:
            first_ecsv.remove_column(col)
            second_ecsv.remove_column(col)
    first_npy = first_ecsv.as_array()
    second_npy = second_ecsv.as_array()
    is_same = True
    if shuffled_lines:
        for line in first_npy:
            if line not in second_npy:
                print(f"SHUFFLE: At line {line}\n from {first_file}\n\n")
                is_same = False
                break
    else:
        comparison = first_npy == second_npy
        is_same = comparison.all()
    return is_same


def compare_line_by_line(first_file, second_file, shuffled_lines=False, line_start=0):
    with open(first_file, encoding="utf-8") as f_1:
        with open(second_file, encoding="utf-8") as f_2:
            f1_lines = f_1.read().splitlines()
            f2_lines = f_2.read().splitlines()
            f1_length = len(f1_lines)
            f2_length = len(f2_lines)
            line_index = 0
            is_same = f1_length == f2_length
            while is_same and (line_index < f1_length):
                if shuffled_lines:
                    is_same = f1_lines[line_index][line_start:] in [
                        f_l[line_start:] for f_l in f2_lines
                    ]
                    if not is_same:
                        print(
                            f"SHUFFLE: At line number {line_index}\n from {first_file}\n{f1_lines[line_index]}\n"
                        )
                else:
                    is_same = (
                        f1_lines[line_index][line_start:]
                        == f2_lines[line_index][line_start:]
                    )
                    if not is_same:
                        print(
                            f"At line number {line_index}\n from {first_file}\n{f1_lines[line_index]}\n from {second_file}\n{f2_lines[line_index]}"
                        )
                line_index += 1
            return is_same
