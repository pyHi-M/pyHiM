#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marcnol, march 25

This script reads a chromatin trace file and a BED file containing genomic coordinates for barcodes.
It assigns genomic coordinates (Chrom, Chrom_Start, Chrom_End) from the BED file to each row
in the trace table based on the 'Barcode #' column.

Usage:
    python trace_impute_genomic_coordinates.py --input trace_file.ecsv --bed bed_file.bed --output output_file.ecsv

Arguments:
    --input  : Path to the input chromatin trace file (ECSV format).
    --bed    : Path to the BED file containing genomic coordinates.
    --output : Path to save the updated trace file (default: appends '_imputed' to the input filename).
"""

import argparse

from matrixOperations.chromatin_trace_table import ChromatinTraceTable


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign genomic coordinates to a chromatin trace table."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input trace file (ECSV format)."
    )
    parser.add_argument(
        "--bed",
        required=True,
        help="Path to the BED file containing genomic coordinates.",
    )
    parser.add_argument("--output", help="Path to save the updated trace file.")
    return parser.parse_args()


def load_bed_file(bed_file):
    """Loads the BED file into a dictionary mapping barcode numbers to genomic coordinates.
    Handles files with inconsistent tab spacing by using regex splitting."""
    bed_dict = {}

    with open(bed_file, "r") as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            # Split on any number of whitespace characters
            # This handles inconsistent tabs/spaces more robustly
            fields = line.strip().split()

            # Ensure we have exactly 4 fields
            if len(fields) != 4:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue

            try:
                chrom = fields[0]
                chrom_start = int(fields[1])
                chrom_end = int(fields[2])
                barcode = int(fields[3])

                bed_dict[barcode] = {
                    "Chrom": chrom,
                    "Chrom_Start": chrom_start,
                    "Chrom_End": chrom_end,
                }
            except ValueError as e:
                print(f"Warning: Skipping line with invalid data types: {line.strip()}")
                print(f"Error: {e}")

    if not bed_dict:
        raise ValueError("No valid entries found in the BED file.")

    print(f"Successfully loaded {len(bed_dict)} barcode mappings from BED file.")
    return bed_dict


def impute_genomic_coordinates(trace_file, bed_dict, output_file):
    """Updates the Chrom, Chrom_Start, and Chrom_End columns in the trace file based on the BED file."""
    trace_table = ChromatinTraceTable()
    trace_table.load(trace_file)

    if trace_table.data is None or len(trace_table.data) == 0:
        print("Error: The trace file is empty or could not be loaded.")
        return

    for row in trace_table.data:
        barcode = row["Barcode #"]
        if barcode in bed_dict:
            row["Chrom"] = bed_dict[barcode]["Chrom"]
            row["Chrom_Start"] = bed_dict[barcode]["Chrom_Start"]
            row["Chrom_End"] = bed_dict[barcode]["Chrom_End"]

    trace_table.save(
        output_file,
        trace_table.data,
        comments="Genomic coordinates imputed from BED file.",
    )
    print(f"Updated trace file saved to {output_file}")


def main():
    args = parse_arguments()
    bed_dict = load_bed_file(args.bed)

    output_file = (
        args.output if args.output else args.input.replace(".ecsv", "_imputed.ecsv")
    )
    impute_genomic_coordinates(args.input, bed_dict, output_file)


if __name__ == "__main__":
    main()
