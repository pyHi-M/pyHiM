#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:49:22 2022

@author: marcnol

converts NPY to TIFF

"""

import os
import select
import sys

import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity
from tifffile import TiffWriter


def main():
    files = sys.argv[1:]

    if select.select(
        [
            sys.stdin,
        ],
        [],
        [],
        0.0,
    )[0]:
        print("Found data in stdin !\n")

        piped_files = [line.rstrip("\n") for line in sys.stdin]
        if len(piped_files) > 0:
            files = piped_files
            # print(f"Read {piped_files} files list from stdin!\n")
    else:
        print("No stdin")

    if len(files) > 0:
        print(f"{len(files)} files to process: ")
        for file in files:
            print(f"\n{os.path.basename(file)}")

            # loads NPY
            data = np.load(file)

            # rescales intensities
            data = rescale_intensity(data, out_range=(0, 1))
            data = equalize_adapthist(data)  # , kernel_size = 2)
            data = data - np.min(data)
            data = data / np.max(data) * (2**14)

            # saves TIFF
            file_out = file.rsplit(".")[0] + ".tif"
            print(f"output file: {file_out}")

            with TiffWriter(file_out) as tif:
                tif.save(data.astype(np.uint16))

            print(f"Finished processing {len(files)} files")

    else:
        print("Please provide list of NPY files to process! none provided")

        print("\n Usage:\n")
        print("with find:")
        print('npy_to_tiff $(find -name "*ch0*_2d_registered.npy")\n\n')
        print("with pipes:")
        print("ls *ch0*_2d_registered.npy | npy_to_tiff")


if __name__ == "__main__":
    main()
