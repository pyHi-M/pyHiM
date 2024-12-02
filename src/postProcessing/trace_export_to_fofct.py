#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a pyHiM trace table in ECSV format to a CSV file fit for the FOF-CT format.
(https://fish-omics-format.readthedocs.io/en/latest/index.html)

We use a JSON file to store the metadata of the experiment, such as:
- the genome assembly
- the experimenter's name
- contact information

required_keys = ["genome_assembly", "experimenter_name", "experimenter_contact"]

Example json file:
{
  "genome_assembly": "GRCh38",
  "experimenter_name": "Dr. Pirulo",
  "experimenter_contact": "pirulo@gmail.com"
}

To link the traces to the chromosomes, we use a BED file that contains the barcode information.
We expect the BED file to have the following columns:
- chrName
- startSeq
- endSeq
- Barcode_ID

The bed file should have no header.

Example BED file:
chr2L	2343645	2356099	5
chr2L	2356147	2369783	9
chr2L	2369828	2381912	13
chr2L	2381947	2393854	17
chr2L	2393892	2405589	21

For this first version, we don't consider, from the pyHiM trace table:
- the "mask_id" column: it's can be linked to the "Cell_ID" column but sometimes it's not the case, two "mask_id" can be linked to the same Cell_ID.
- the "label" column: usually it's linked to RNA species but it's like a global mask for many cells, so it's not RNA spots.

The output CSV file will have the following columns:
- Spot_ID
- Trace_ID
- X
- Y
- Z
- Chrom
- Chrom_Start
- Chrom_End
- Extra_Cell_ROI_ID ("ROI #" in the pyHiM trace table)

We need as run arguments:
- the path to the ECSV file
- the path to the BED file
- the path to the JSON file (optional)
- the path to the output CSV file (optional)

Example:
python export_trace_table.py --ecsv_file /path/to/Trace_3D_barcode_KDtree_ROI-5.ecsv --bed_file /path/to/barcode.bed --json_file /path/to/parameters.json --output_file /path/to/output.csv

"""

import csv
import json
import os
from argparse import ArgumentParser

import pandas as pd
from astropy.io import ascii


def parse_args():
    parser = ArgumentParser(
        description="Convert a pyHiM trace table to a FOF-CT CSV file"
    )
    parser.add_argument("--ecsv_file", help="Path to the ECSV file")
    parser.add_argument("--bed_file", help="Path to the BED file")
    parser.add_argument(
        "--json_file",
        default=os.getcwd() + os.sep + "parameters.json",
        help="Path to the JSON file with the metadata. Default: parameters.json",
    )
    # Optional argument
    parser.add_argument(
        "--output_file", default=None, help="Path to the output CSV file"
    )
    return parser.parse_args()


def load_trace_ecsv_file(ecsv_file):
    # Load the ECSV file
    ecsv_data = ascii.read(ecsv_file, format="ecsv")
    print(f"Trace table loaded from '{ecsv_file}'")
    return ecsv_data


def load_barcode_bed_file(bed_file):
    # Define the column names for a BED file
    column_names = [
        "chrName",
        "startSeq",
        "endSeq",
        "Barcode_ID",
    ]
    # Load the BED file
    bed_data = pd.read_csv(bed_file, sep="\t", names=column_names, comment="#")
    print(f"BED file loaded from '{bed_file}'")
    return bed_data


def load_json_file(json_file):
    with open(json_file, "r") as f:
        metadata = json.load(f)
    print(f"Metadata loaded from '{json_file}'")
    return metadata


def check_metadata(metadata):
    # Check if the metadata contains the required keys
    required_keys = ["genome_assembly", "experimenter_name", "experimenter_contact"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Metadata JSON file is missing the key '{key}'")


def get_output_file(ecsv_file, output_file):
    # Check if the output file was provided
    if output_file is not None:
        if ~os.path.exists(output_file):
            return output_file
        else:
            print(
                f"[WARNING] Output file '{output_file}' already exists, using default name."
            )
    basename = os.path.basename(ecsv_file).split(".")[0]
    return os.getcwd() + os.sep + basename + "_FOFCT" + ".csv"


def find_chrom_info(bed_data, barcode_id):
    # Find the row with the given barcode ID
    barcode_row = bed_data[bed_data["Barcode_ID"] == barcode_id]

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
        barcode_id = row["Barcode #"]

        # Find the chromosome information
        chrom, start, end = find_chrom_info(bed_data, barcode_id)

        # Assign the chromosome information to the new columns
        ecsv_data["Chrom"][i] = chrom
        ecsv_data["Chrom_Start"][i] = start
        ecsv_data["Chrom_End"][i] = end

    return ecsv_data


def remove_unused_columns(ecsv_data):
    # Remove unused columns
    ecsv_data.remove_columns(["Barcode #", "Mask_id", "label"])
    return ecsv_data


def rename_pyhim_columns(ecsv_data):
    # Rename columns
    ecsv_data.rename_column("x", "X")
    ecsv_data.rename_column("y", "Y")
    ecsv_data.rename_column("z", "Z")
    ecsv_data.rename_column("ROI #", "Extra_Cell_ROI_ID")
    return ecsv_data


def write_fof_ct_csv(csv_file, header, ecsv_data):
    # Open the CSV file for writing
    with open(csv_file, "w", newline="") as f:
        f.write(header)

        writer = csv.writer(f)
        # Write the data
        for row in ecsv_data:
            writer.writerow(row)
    print(f"CSV file written to '{csv_file}'")


def get_initials(first_name):
    initials = ""
    for name in first_name.split("-"):
        if initials == "":
            initials += name[0]
        else:
            initials += "-" + name[0]
    return initials


def get_authors_from_copyright():
    copyright_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "COPYRIGHT.txt"
    )
    with open(copyright_path, "r") as f:
        authors = f.read().splitlines()
        authors_formatting = ""
        for line in authors:
            elt = line.split(",")
            if len(elt) == 2:
                author = elt[0].split(" ")
                first_name = author[0]
                last_name = author[1]
                initials = get_initials(first_name)
                authors_formatting = f"{last_name}, {initials};"

    return authors_formatting


def get_header_comments(genome_assembly, experimenter_name, experimenter_contact):
    header = ""
    header += "##FOF-CT_version=v0.1\n"
    header += "##Table_namespace=4dn_FOF-CT_core\n"
    header += f"##genome_assembly={genome_assembly}\n"
    header += "##XYZ_unit=micron\n"
    header += "#Software_Title: pyHiM\n"
    header += "#Software_Type: SpotLoc+Tracing\n"  # TODO: Check this
    authors_formatting = get_authors_from_copyright()
    header += f"#Software_Authors: {authors_formatting}\n"
    header += "#Software_Description: pyHiM implements the analysis of multiplexed DNA-FISH data.\n"
    header += "#Software_Repository: https://github.com/marcnol/pyHiM\n"
    header += (
        "#Software_PreferredCitationID: https://doi.org/10.1186/s13059-024-03178-x\n"
    )
    header += "#lab_name: Nollmann Lab\n"
    header += f"#experimenter_name: {experimenter_name}\n"
    header += f"#experimenter_contact: {experimenter_contact}\n"
    header += "#additional_tables:\n"
    header += "##columns=(Spot_ID, Trace_ID, X, Y, Z, Chrom, Chrom_Start, Chrom_End, Extra_Cell_ROI_ID)\n"
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
    ecsv_data = remove_unused_columns(ecsv_data)
    # Rename columns
    ecsv_data = rename_pyhim_columns(ecsv_data)
    # Generate header comments
    header = get_header_comments(
        genome_assembly, experimenter_name, experimenter_contact
    )
    # Write the CSV file
    write_fof_ct_csv(csv_file, header, ecsv_data)


if __name__ == "__main__":
    args = parse_args()

    # Load the metadata
    metadata = load_json_file(args.json_file)
    check_metadata(metadata)

    output = get_output_file(args.ecsv_file, args.output_file)

    convert_ecsv_to_csv(
        args.ecsv_file,
        output,
        args.bed_file,
        metadata["genome_assembly"],
        metadata["experimenter_name"],
        metadata["experimenter_contact"],
    )
