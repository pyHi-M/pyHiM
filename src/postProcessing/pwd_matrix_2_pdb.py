#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:27:01 2023

@author: marcnol

from a set of coordinates it calculates the PWD matrix, and from it it gets back the coordinates.


"""
import argparse
import os
import select
import sys

import numpy as np

from core.data_manager import create_folder
from core.parameters import loads_barcode_dict
from matrixOperations.HIMmatrixOperations import (
    calculate_ensemble_pwd_matrix,
    distances_2_coordinates,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Name of input matrix files in NPY.")
    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )
    parser.add_argument(
        "--barcode_type_dict",
        help="Json dictionary linking barcodes and atom types (MUST BE 3 characters long!). ",
    )
    p = {}

    args = parser.parse_args()
    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.barcode_type_dict:
        p["barcode_type_dict"] = args.barcode_type_dict
    else:
        p["barcode_type_dict"] = "barcode_type_dict.json"

    p["matrix_files"] = []
    if args.pipe:
        p["pipe"] = True
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.0,
        )[0]:
            p["matrix_files"] = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        p["pipe"] = False
        p["matrix_files"] = [p["input"]]

    return p


def xyz_2_pdb(file_name, xyz, barcode_type=dict()):
    n_atoms = xyz.shape[0]
    barcodes = [x for x in range(n_atoms)]
    default_atom_name = "xxx"
    trace_name = "avr"

    if len(barcode_type) < 1:
        # all atoms have the same identity
        print("did not find barcode_type dictionary")
        for i, barcode in enumerate(barcodes):
            barcode_type["{}".format(barcode)] = default_atom_name
    else:
        # adds missing keys
        for barcode in barcodes:
            if str(barcode) not in barcode_type.keys():
                barcode_type["{}".format(barcode)] = default_atom_name
                print("$ fixing key {} as not found in dict".format(barcode))

    with open(file_name, mode="w+", encoding="utf-8") as fid:
        # atom coordinates
        field_record = "HETATM"
        field_atom_number = " {: 4d} "
        field_atom_name = " C{:02d}"
        field_alternative_location_indicator = " "
        field_res_name = trace_name + " "
        field_chain_identifier = " "
        field_residue_seq_number = "{:4d}"
        field_code_insertion = "    "
        field_X = " {:0<7.3f}"
        field_Y = " {:0<7.3f}"
        field_Z = " {:0<7.3f}"
        field_occupancy = "   0.0"
        field_temp_factor = "   0.0"
        field_segment_identifier = "      " + "PSDO"
        field_element_symbol = " X"
        field_charge_atom = " X"
        txt = (
            field_record
            + field_atom_number
            + field_atom_name
            + field_alternative_location_indicator
            + field_res_name
            + field_chain_identifier
            + field_residue_seq_number
            + field_code_insertion
            + field_X
            + field_Y
            + field_Z
            + field_occupancy
            + field_temp_factor
            + field_segment_identifier
            + field_element_symbol
            + field_charge_atom
            + "\n"
        )

        # txt = "HETATM  {: 3d}  C{:02d} {} P   1      {: 5.3f}  {: 5.3f}  {: 5.3f}  0.00  0.00      PSDO C  \n"
        for i in range(n_atoms):
            fid.write(
                txt.format(
                    i + 1, i + 1, int(barcodes[i]), xyz[i, 0], xyz[i, 1], xyz[i, 2]
                )
            )

        # connectivity
        txt1 = "CONNECT  {: 3d}  {: 3d}\n"
        txt2 = "CONNECT  {: 3d}  {: 3d}  {: 3d}\n"

        # first line of connectivity
        fid.write(txt1.format(1, 2))

        # consecutive lines
        for i in range(2, n_atoms):
            fid.write(txt2.format(i, i - 1, i + 1))

        # last line
        fid.write(txt1.format(i + 1, i))

        print("Done writing {:s} with {:d} atoms.".format(file_name, n_atoms))


def remove_nans(ensemble_matrix, min_number_nans=3):
    ensemble_matrix_no_nans = ensemble_matrix.copy()

    # nan_matrix = np.isnan(ensemble_matrix_no_nans)

    # Find columns with all NaN values
    # nan_columns = np.all(np.isnan(ensemble_matrix_no_nans), axis=0)
    # cols_with_nans = np.where(nan_columns)[0]

    nan_counts = np.sum(np.isnan(ensemble_matrix_no_nans), axis=0)

    cols_with_nans = np.where(nan_counts > min_number_nans)[0]

    ensemble_matrix_no_nans = np.delete(ensemble_matrix_no_nans, cols_with_nans, axis=0)
    ensemble_matrix_no_nans = np.delete(ensemble_matrix_no_nans, cols_with_nans, axis=1)

    np.fill_diagonal(ensemble_matrix_no_nans, 0)

    return ensemble_matrix_no_nans


def matrix_2_pdb(
    sc_matrix, folder_path, barcode_type=dict(), output_file="ensemble_pwd_matrix"
):
    # gets ensemble matrix from list of single matrices
    pixel_size = 1
    cells_to_plot = range(sc_matrix.shape[2])
    ensemble_matrix, _ = calculate_ensemble_pwd_matrix(
        sc_matrix, pixel_size, cells_to_plot, mode="median"
    )
    ensemble_matrix = remove_nans(ensemble_matrix, min_number_nans=3)

    np.save(folder_path + os.sep + output_file + ".npy", ensemble_matrix)
    # converts ensemble matrix to coordinates
    coords = distances_2_coordinates(ensemble_matrix)

    # converts to PDB and outputs
    xyz_2_pdb(
        folder_path + os.sep + output_file + ".pdb", coords, barcode_type=barcode_type
    )


def runtime(
    matrix_files=[],
    folder_path="./ensemble_structure",
    barcode_type=dict(),
):
    if len(matrix_files) > 0:
        for matrix_file in matrix_files:
            if os.path.exists(matrix_file):
                sc_matrix = np.load(matrix_file)

                matrix_2_pdb(sc_matrix, folder_path, barcode_type=barcode_type)

            else:
                print("! ERROR: could not find {}".format(matrix_file))
    else:
        print("! ERROR: no matrix file provided.")
    return len(matrix_files)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main function"""

    # [parsing arguments]
    p = parse_arguments()
    barcode_type = loads_barcode_dict(p["barcode_type_dict"])

    # creates output folder
    output_folder = "ensemble_structure"
    folder_path = os.path.join(
        os.getcwd(), output_folder
    )  # Specify the folder path here

    create_folder(folder_path)

    n_traces_processed = runtime(
        matrix_files=p["matrix_files"],
        folder_path=folder_path,
        barcode_type=barcode_type,
    )

    print(f"Processed <{n_traces_processed}> trace file(s)")
    print("Finished execution")


if __name__ == "__main__":
    main()


"""
    distances = coord_2_distances(coords)
    rec_coords = distances_2_coordinates(distances)

    visualize([
        (coords, 'original points'),
        (rec_coords, 'reconstructed points')
    ])

    plt.imshow( distances, cmap='Reds')
    plt.colorbar()
    plt.show()

    rec_PWD = coord_2_distances(rec_coords)
    plt.imshow(rec_PWD, cmap='Reds')
    plt.colorbar()
    plt.show()
"""
