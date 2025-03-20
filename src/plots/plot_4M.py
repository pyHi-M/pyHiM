#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

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
