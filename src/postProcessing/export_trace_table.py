#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import ascii
import csv
import pandas as pd


def load_trace_ecsv_file(ecsv_file):
    # Load the ECSV file
    ecsv_data = ascii.read(ecsv_file, format="ecsv")

    return ecsv_data


def load_barcode_bed_file(bed_file):
    # Define the column names for a BED file
    column_names = [
        "chrName",
        "startSeq",
        "endSeq",
        "barcodeName",
        "numberOligos",
        "genomicSize",
    ]

    # Load the BED file
    bed_data = pd.read_csv(bed_file, sep="\t", names=column_names, comment="#")

    bed_data = bed_data[["chrName", "startSeq", "endSeq", "barcodeName"]]

    return bed_data


def find_chrom_info(bed_data, barcode_id):
    # Find the row with the given barcode ID
    barcode_row = bed_data[bed_data["barcodeName"] == barcode_id]

    # Check if the barcode ID was found
    if len(barcode_row) == 0:
        raise ValueError(f"Barcode ID '{barcode_id}' not found in the BED file")

    # Extract the chromosome and start position
    chrom = barcode_row["chrName"].values[0]
    start = barcode_row["startSeq"].values[0]
    end = barcode_row["endSeq"].values[0]

    return chrom, start, end


def assign_chrom_info(ecsv_data, bed_data):
    # Loop over the rows in the ECSV data
    for i, row in enumerate(ecsv_data):
        # Extract the barcode ID
        barcode_id = row["Barcode_ID"]

        # Find the chromosome information
        chrom, start, end = find_chrom_info(bed_data, barcode_id)

        # Assign the chromosome information to the new columns
        ecsv_data["Chrom"][i] = chrom
        ecsv_data["Chrom_Start"][i] = start
        ecsv_data["Chrom_End"][i] = end

    return ecsv_data


def get_header_comments(genome_assembly, experimenter_name, experimenter_contact):
    header = ""
    header += "##FOF-CT_version=v0.1\n"
    header += "##Table_namespace=4dn_FOF-CT_core\n"
    header += f"##genome_assembly={genome_assembly}\n"
    header += "##XYZ_unit=micron\n"
    header += "#Software_Title: pyHiM\n"
    header += "#Software_Type: SpotLoc+Tracing\n"  # TODO: Check this
    header += "#Software_Authors: Nollmann, M; Fiche, J-B; Goetz, M; Devos, X\n"
    header += "#Software_Description: pyHiM implements the analysis of multiplexed DNA-FISH data.\n"
    header += "#Software_Repository: https://github.com/marcnol/pyHiM\n"
    header += (
        "#Software_PreferredCitationID: https://doi.org/10.1186/s13059-024-03178-x\n"
    )
    header += "#lab_name: Nollmann Lab\n"
    header += f"#experimenter_name: {experimenter_name}\n"
    header += f"#experimenter_contact: {experimenter_contact}\n"
    header += "#additional_tables:\n"
    header += "##columns=(Spot_ID, Trace_ID, X, Y, Z, Chrom, Chrom_Start, Chrom_End, Cell_ID, Extra_Cell_ROI_ID)\n"
    return header


def convert_ecsv_to_csv(
    ecsv_file,
    csv_file,
    bed_file,
    genome_assembly,
    experimenter_name,
    experimenter_contact,
):
    # Load files
    ecsv_data = load_trace_ecsv_file(ecsv_file)
    bed_data = load_barcode_bed_file(bed_file)
    # Assign chromosome information
    ecsv_data = assign_chrom_info(ecsv_data, bed_data)
    # Remove unused columns
    ecsv_data.remove_columns(["Barcode #", "label"])
    # Rename columns
    ecsv_data.rename_column("x", "X")
    ecsv_data.rename_column("y", "Y")
    ecsv_data.rename_column("z", "Z")
    ecsv_data.rename_column("ROI #", "Extra_Cell_ROI_ID")
    ecsv_data.rename_column("Mask_id", "Cell_ID")
    # Generate header comments
    header = get_header_comments(
        genome_assembly, experimenter_name, experimenter_contact
    )

    # Open the CSV file for writing
    with open(csv_file, "w", newline="") as f:
        f.write(header)

        writer = csv.writer(f)
        # Write the data
        for row in ecsv_data:
            writer.writerow(row)


# # Use the function
# path = "/home/xdevos/Repositories/marcnol/pyHiM/pyhim-small-dataset/resources/traces_dataset/OUT/build_traces/data/"
# convert_ecsv_to_csv(path + "Trace_3D_barcode_KDtree_ROI-5.ecsv", path + "output.csv")
