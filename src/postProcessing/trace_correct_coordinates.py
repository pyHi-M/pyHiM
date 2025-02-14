#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script corrects chromatin trace localization errors by adjusting the z-coordinate of barcodes
to minimize their deviation from the center of mass (CoM) of their respective traces.

Usage:
    python trace_correct_coordinates.py --input traces.ecsv --output traces_corrected.ecsv --max_iter 10 --tolerance 0.01

Algorithm:
    1) Compute the initial center of mass (CoM) for each chromatin trace.
    2) Iteratively adjust the z-coordinates of each barcode to minimize its distance to the CoM.
    3) Update the trace table and save the corrected version.

Arguments:
    --input      Path to the input chromatin trace table (ECSV format).
    --output     Path to save the corrected trace table (Default: appends "_corrected").
    --max_iter   Maximum number of iterations (default: 10).
    --tolerance  Convergence criterion: Stop if all barcode shifts are below this threshold (default: 0.01).

Dependencies:
    - numpy
    - astropy.table
"""

import argparse
import os

import numpy as np
from astropy.table import Table


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Correct z-offsets for chromatin traces."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input trace file (ECSV format)."
    )
    parser.add_argument(
        "--output",
        help="Path to save the corrected trace file. Default: appends '_corrected'.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Convergence threshold for Z shifts (default: 0.01).",
    )
    return parser.parse_args()


def compute_center_of_mass(trace_table):
    """
    Compute the center of mass (CoM) for each chromatin trace.

    Parameters:
    ----------
    trace_table : astropy.table.Table
        Input chromatin trace table.

    Returns:
    -------
    dict
        A dictionary mapping each Trace_ID to its (x, y, z) center of mass.
    """
    com_dict = {}
    trace_table_by_id = trace_table.group_by("Trace_ID")

    for sub_trace in trace_table_by_id.groups:
        trace_id = sub_trace["Trace_ID"][0]
        com_x = np.mean(sub_trace["x"])
        com_y = np.mean(sub_trace["y"])
        com_z = np.mean(sub_trace["z"])
        com_dict[trace_id] = (com_x, com_y, com_z)

    return com_dict


def optimize_z_offsets(trace_table, max_iter=10, tolerance=0.01):
    """
    Iteratively correct Z-offsets by optimizing barcode positions.

    Parameters:
    ----------
    trace_table : astropy.table.Table
        Input chromatin trace table.
    max_iter : int
        Maximum number of iterations.
    tolerance : float
        Convergence threshold for stopping criterion.

    Returns:
    -------
    astropy.table.Table
        Corrected trace table with updated Z-coordinates.
    """
    new_trace_table = Table(trace_table)  # Copy input table for modification
    unique_barcodes = np.unique(trace_table["Barcode #"])

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}...")

        com_dict = compute_center_of_mass(new_trace_table)  # Compute updated CoMs
        z_shifts = np.zeros(len(unique_barcodes))  # Track shifts for convergence check

        for i, barcode in enumerate(unique_barcodes):
            barcode_indices = np.where(new_trace_table["Barcode #"] == barcode)[0]

            if len(barcode_indices) == 0:
                continue

            # Extract Z-coordinates and corresponding trace CoMs
            barcode_z_values = new_trace_table["z"][barcode_indices]
            trace_ids = new_trace_table["Trace_ID"][barcode_indices]

            com_z_values = np.array([com_dict[trace_id][2] for trace_id in trace_ids])

            # Compute optimal shift as the mean difference
            z_shift = np.mean(com_z_values - barcode_z_values)
            z_shifts[i] = abs(z_shift)  # Track shift magnitude for convergence check

            # Apply the shift to the barcode across all traces
            new_trace_table["z"][barcode_indices] += z_shift

        max_shift = np.max(z_shifts)
        print(f"Max Z shift in iteration: {max_shift:.4f}")

        if max_shift < tolerance:
            print("Convergence reached!")
            break

    return new_trace_table


def main():
    args = parse_arguments()

    # Determine output filename
    if args.output:
        output_filename = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_filename = f"{base}_corrected{ext}"

    # Load the trace table
    print(f"Loading trace table: {args.input}")
    trace_table = Table.read(args.input, format="ascii.ecsv")

    # Apply Z-offset correction
    print(
        f"Optimizing Z-offsets with max {args.max_iter} iterations and tolerance {args.tolerance}..."
    )
    corrected_trace_table = optimize_z_offsets(
        trace_table, args.max_iter, args.tolerance
    )

    # Save the corrected trace table
    corrected_trace_table.write(output_filename, format="ascii.ecsv", overwrite=True)
    print(f"Saved corrected trace table: {output_filename}")


if __name__ == "__main__":
    main()
