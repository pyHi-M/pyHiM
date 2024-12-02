#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:56:27 2020

@author: marcnol

make links of files in a directory to a second directory

In the command line, run as

Example:

$ lndir.py "/home/marcnol/Repositories/pyHiM/*py" ~/Downloads/test

Make sure that the first argument has quotation marks if you use wildcards!

"""

import glob
import os
import sys

from core.pyhim_logging import write_string_to_file

# =============================================================================
# MAIN
# =============================================================================


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Not enough arguments")

    file_list_string = sys.argv[1]
    dest_folder = sys.argv[2]

    print(f"file_list = {file_list_string} | destDir = {dest_folder}")

    file_list = list(glob.glob(file_list_string))

    if len(file_list) > 0:
        file_name = os.path.dirname(file_list[0]) + os.sep + "lndir.log"

        for file in file_list:
            new_file = dest_folder + os.sep + os.path.basename(file)
            print(f"{file}-->{new_file}")

            command = "ln -s " + file + " " + new_file
            os.system(command)

            write_string_to_file(file_name, command, attribute="a")

        print(
            f"Linked {len(file_list)} files form {os.path.dirname(file_list[0])} to {dest_folder}"
        )

    else:
        print("File List is empty :(")


if __name__ == "__main__":
    main()
