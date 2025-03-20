#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mar 20 2025

@author: marcnol

plot_4M.py
==========

Description:
-----------
A script for performing 4M (Multi-way Measurement of Molecular interactions in space) analysis
on chromatin trace data. This tool analyzes spatial colocalization between a specified anchor
barcode and all other barcodes in 3D chromatin trace data.

The script calculates colocalization frequencies based on a distance cutoff and performs
bootstrapping to estimate statistical confidence (mean and standard error). It generates
a plot showing the frequency of interaction between the anchor barcode and all others.

This is particularly useful for analyzing chromatin organization, DNA-DNA interactions,
and spatial proximity relationships in microscopy data.

Usage:
-----
    $ python plot_4M.py --input TRACE_FILE.ecsv --anchor BARCODE_NUMBER [options]
    $ cat file_list.txt | python plot_4M.py --pipe --anchor BARCODE_NUMBER [options]
    $ find . -name "*.ecsv" | python plot_4M.py --pipe --anchor BARCODE_NUMBER [options]

Arguments:
---------
    --input TRACE_FILE          Path to input trace table in ECSV format
    --anchor BARCODE_NUMBER     Anchor barcode number for colocalization analysis
    --cutoff DISTANCE           Distance cutoff for colocalization (default: 0.2 µm)
    --bootstrapping_cycles N    Number of bootstrap iterations (default: 10)
    --output FILENAME           Output file name for the plot (default: colocalization_plot.png)
    --pipe                      Read trace file list from stdin (for batch processing)

Examples:
--------
1. Analyze a single trace file with default parameters:
   $ python plot_4M.py --input traces.ecsv --anchor 42

2. Analyze with custom distance cutoff and more bootstrap cycles:
   $ python plot_4M.py --input traces.ecsv --anchor 42 --cutoff 0.25 --bootstrapping_cycles 100

3. Process multiple trace files in batch mode:
   $ cat trace_files.txt | python plot_4M.py --pipe --anchor 42 --output batch_results.png

4. Process all ECSV files in a directory:
   $ find ./data -name "*.ecsv" | python plot_4M.py --pipe --anchor 42

Output:
------
- A PNG image with a plot showing colocalization frequencies between the anchor barcode
  and all other barcodes, including error bars derived from bootstrapping
- The output filename will be modified to include the anchor barcode number
  (e.g., "colocalization_plot_anchor_42.png")

Notes:
-----
- The script requires the ChromatinTraceTable class from the matrixOperations module
- Input files must be in ECSV format compatible with ChromatinTraceTable.load()
- Bootstrapping is used to estimate mean and standard error of colocalization frequencies
- The distance cutoff is in micrometers (µm)
- Higher numbers of bootstrapping cycles increase statistical confidence but require more computation time
"""
import argparse
import select
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from matrixOperations.chromatin_trace_table import ChromatinTraceTable


def compute_colocalization(trace_table, anchor_barcode, distance_cutoff):
    """Computes the frequency of colocalization between the anchor barcode and all other barcodes."""
    barcode_interactions = {}
    trace_groups = trace_table.group_by("Trace_ID").groups

    for trace in tqdm(trace_groups, desc="Processing traces"):
        anchor_positions = trace[trace["Barcode #"] == anchor_barcode]
        other_barcodes = np.unique(trace["Barcode #"])

        for barcode in other_barcodes:
            if barcode == anchor_barcode:
                continue

            other_positions = trace[trace["Barcode #"] == barcode]

            if len(anchor_positions) > 0 and len(other_positions) > 0:
                distances = np.linalg.norm(
                    np.array(
                        [
                            anchor_positions["x"],
                            anchor_positions["y"],
                            anchor_positions["z"],
                        ]
                    ).T[:, None]
                    - np.array(
                        [
                            other_positions["x"],
                            other_positions["y"],
                            other_positions["z"],
                        ]
                    ).T,
                    axis=-1,
                )

                colocalized = np.any(distances < distance_cutoff)
                if barcode not in barcode_interactions:
                    barcode_interactions[barcode] = [0, 0]

                barcode_interactions[barcode][1] += 1  # Total occurrences
                if colocalized:
                    barcode_interactions[barcode][0] += 1  # Co-localized occurrences

    # Compute frequencies
    barcode_frequencies = {
        barcode: (count[0] / count[1] if count[1] > 0 else 0)
        for barcode, count in barcode_interactions.items()
    }
    barcode_frequencies[anchor_barcode] = 0.0
    return barcode_frequencies


def bootstrap_colocalization(
    trace_table, anchor_barcode, distance_cutoff, n_bootstrap=100
):
    """Performs bootstrapping to estimate mean and SEM of colocalization frequencies."""
    barcode_samples = {}
    trace_ids = np.unique(trace_table["Trace_ID"])

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        sampled_traces = np.random.choice(trace_ids, size=len(trace_ids), replace=True)
        sampled_table = trace_table[np.isin(trace_table["Trace_ID"], sampled_traces)]

        colocalization = compute_colocalization(
            sampled_table, anchor_barcode, distance_cutoff
        )

        for barcode, frequency in colocalization.items():
            if barcode not in barcode_samples:
                barcode_samples[barcode] = []
            barcode_samples[barcode].append(frequency)

    barcode_means = {
        barcode: np.mean(values) for barcode, values in barcode_samples.items()
    }
    barcode_sems = {
        barcode: np.std(values) / np.sqrt(n_bootstrap)
        for barcode, values in barcode_samples.items()
    }
    return barcode_means, barcode_sems


def plot_colocalization(barcode_means, barcode_sems, anchor, output_file):
    """Plots the colocalization frequencies with error bars."""
    barcodes = sorted(barcode_means.keys())
    print(barcodes)

    means = [barcode_means[b] for b in barcodes]
    errors = [barcode_sems[b] for b in barcodes]

    plt.figure(figsize=(10, 5))
    plt.errorbar(barcodes, means, yerr=errors, fmt="o-", capsize=5)
    plt.axvline(
        anchor, color="red", linestyle="--", label="Anchor Barcode"
    )  # Highlight anchor
    plt.xlabel("Barcode #", fontsize=13)
    plt.ylabel("Colocalization frequency", fontsize=13)
    plt.title(f"4M plot for anchor: {str(anchor)}", fontsize=15)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(True)

    output_file = f"{output_file.split('.')[0]}_anchor_{str(anchor)}.png"
    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute barcode interaction frequencies with bootstrapping."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input trace table (ECSV format)."
    )
    parser.add_argument(
        "--anchor", type=int, required=True, help="Anchor barcode number."
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        required=False,
        default=0.2,
        help="Distance cutoff for colocalization. Default = 0.2 um",
    )
    parser.add_argument(
        "--bootstrapping_cycles",
        type=int,
        default=10,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--output", default="colocalization_plot.png", help="Output file for the plot."
    )
    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )
    args = parser.parse_args()

    trace_files = []
    if args.pipe:
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.0,
        )[0]:
            trace_files = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        trace_files = [args.input]

    if len(trace_files) > 0:

        for trace_file in trace_files:

            trace = ChromatinTraceTable()
            trace.initialize()
            trace.load(trace_file)

            barcode_means, barcode_sems = bootstrap_colocalization(
                trace.data, args.anchor, args.cutoff, args.bootstrapping_cycles
            )
            plot_colocalization(barcode_means, barcode_sems, args.anchor, args.output)

    else:
        print("\nNo trace files were detected")


if __name__ == "__main__":
    main()
