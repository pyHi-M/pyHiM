#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts a FOF-CT CSV file back to a pyHiM trace table in ECSV format.

Required inputs:
- The path to the FOFCT file in CSV format containing the trace data.
- The path to the BED file containing barcode information.
- The path to a JSON file with metadata (optional).

The script will produce an ECSV file that restores the missing columns (`Barcode #`, `Mask_id`, and `label`).

# Arguments:
- `--csv_file`: (str) Path to the CSV file that contains the trace data.
- `--bed_file`: (str) Path to the BED file that contains the chromosome and barcode information.
- `--json_file`: (str) Path to the JSON file with metadata, such as genome assembly and experimenter information. (Optional; default: `parameters.json` in the current working directory).
- `--output_file`: (str) Path to the output ECSV file. (Optional; default: replaces the `.csv` extension of the input file with `.ecsv`).

# Usage Example:
To convert a CSV file back to the ECSV format using the specified BED and JSON files, you would run:

python convert_csv_to_ecsv.py
  --fofct_file /path/to/output.csv
  --bed_file /path/to/barcode.bed
  --output_file /path/to/Trace_3D_barcode_KDtree_ROI-5.ecsv

If the `--output_file` argument is not provided, the script will save the ECSV file with the same name as the input CSV file but with an `.ecsv` extension.

marcnol, Aug 8 2024
"""

from argparse import ArgumentParser

import pandas as pd
from astropy.io import ascii
from astropy.table import Table


def parse_args():
    parser = ArgumentParser(
        description="Convert a FOF-CT CSV file back to a pyHiM trace table in ECSV format"
    )
    parser.add_argument("--fofct_file", help="Path to the FOFCT file", required=True)
    parser.add_argument("--bed_file", help="Path to the BED file", required=True)
    parser.add_argument(
        "--output_file", default=None, help="Path to the output ECSV file"
    )
    return parser.parse_args()


def read_column_names_from_csv(csv_file):
    # Read the file line by line until finding the `##columns` line
    with open(csv_file, "r") as file:
        for line in file:
            if line.startswith("##columns"):
                # Extract column names, strip leading spaces and return them
                column_names = line.split("=")[1].strip().strip("()").split(", ")
                return [col.strip() for col in column_names]
    raise ValueError("No `##columns` line found in the FOFCT file.")


def load_csv_file(csv_file, column_names):
    # Load the CSV file using pandas, assigning the correct column names
    data = pd.read_csv(csv_file, comment="#", header=None, names=column_names)
    print(f"FOFCT file loaded from '{csv_file}'")
    print(
        f"Columns in FOFCT file: {list(data.columns)}"
    )  # Print out the columns for debugging
    return data


def load_barcode_bed_file(bed_file):
    # Load the BED file
    column_names = ["chrName", "startSeq", "endSeq", "Barcode_ID"]
    bed_data = pd.read_csv(bed_file, sep="\t", names=column_names, comment="#")
    print(f"BED file loaded from '{bed_file}'")
    return bed_data


def validate_columns(data):
    required_columns = ["Chrom", "Chrom_Start", "Chrom_End"]
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(
                f"Required column '{col}' not found in FOFCT file. Please check the input file."
            )
    print("All required columns are present in the FOFCT file.")


def find_barcode_id(bed_data, chrom, chrom_start, chrom_end):
    # Match the chromosome and start/end positions to find the Barcode ID
    barcode_row = bed_data[
        (bed_data["chrName"] == chrom)
        & (bed_data["startSeq"] == chrom_start)
        & (bed_data["endSeq"] == chrom_end)
    ]
    if len(barcode_row) == 0:
        raise ValueError(
            f"Chromosome information not found in the BED file for {chrom}:{chrom_start}-{chrom_end}"
        )
    return barcode_row["Barcode_ID"].values[0]


def add_missing_columns(data, bed_data):
    # Validate that all necessary columns are present in the data
    validate_columns(data)

    # Add Barcode #, Mask_id, and label columns
    barcode_ids = []
    for index, row in data.iterrows():
        try:
            barcode_id = find_barcode_id(
                bed_data, row["Chrom"], row["Chrom_Start"], row["Chrom_End"]
            )
            barcode_ids.append(barcode_id)
        except ValueError as e:
            print(f"Error processing row {index}: {e}")
            barcode_ids.append(None)  # Append None if not found

    data["Barcode #"] = barcode_ids
    data["Mask_id"] = -1  # Placeholder for Mask_id (customize as needed)
    data["label"] = "None"  # Placeholder for label (customize as needed)

    return data


def rename_columns_for_ecsv(data):
    # Rename columns back to their original names
    data.rename(
        columns={"X": "x", "Y": "y", "Z": "z", "Extra_Cell_ROI_ID": "ROI #"},
        inplace=True,
    )
    return data


def convert_csv_to_ecsv(csv_data, output_file):
    # Convert the DataFrame to an astropy Table
    table = Table.from_pandas(csv_data)
    # Save the table as an ECSV file
    ascii.write(table, output_file, format="ecsv", overwrite=True)
    print(f"ECSV file written to '{output_file}'")


def main():
    args = parse_args()

    # Read column names from the CSV file
    column_names = read_column_names_from_csv(args.fofct_file)

    # Load the files
    csv_data = load_csv_file(args.fofct_file, column_names)
    bed_data = load_barcode_bed_file(args.bed_file)

    # Add missing columns
    csv_data = add_missing_columns(csv_data, bed_data)

    # Rename columns to match ECSV format
    csv_data = rename_columns_for_ecsv(csv_data)

    # Define the output ECSV file path
    output_file = (
        args.output_file
        if args.output_file
        else args.fofct_file.replace(".csv", ".ecsv")
    )

    # Convert to ECSV format and save
    convert_csv_to_ecsv(csv_data, output_file)


if __name__ == "__main__":
    main()
