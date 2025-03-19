#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes a chromatin trace table and applies K-means clustering
to split traces when their radius of gyration exceeds a threshold.

Usage:
    python split_traces.py --input original_traces.ecsv --std_threshold 1.5 --num_clusters 3

Example:
    Given a chromatin trace table, this script:
      - Computes radius of gyration (Rg) for all traces.
      - Identifies traces with Rg larger than `mean + N * std_dev`.
      - Uses K-means to split these traces into `num_clusters`.
      - Saves the modified trace table with updated Trace_IDs.

Arguments:
    --input          Path to the input chromatin trace table (ECSV format).
    --output         (Optional) Path to save the modified trace file. Default: appends "_split".
    --std_threshold  Number of std deviations above mean Rg to split traces (default: 1.0).
    --num_clusters   Number of clusters for K-means (default: 2).

Dependencies:
    - numpy
    - sklearn.cluster (KMeans)
    - matrixOperations.chromatin_trace_table (for trace table management)
    - uuid (for unique Trace_ID generation)
"""

import argparse
import os
import select
import sys
import uuid

import numpy as np
from sklearn.cluster import KMeans

from matrixOperations.chromatin_trace_table import (  # Import the class
    ChromatinTraceTable,
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split chromatin traces using K-means clustering."
    )
    parser.add_argument("--input", help="Path to the input trace file.")
    parser.add_argument(
        "--output",
        help="Path to save the modified trace file. Default: appends '_split'.",
    )
    parser.add_argument(
        "--std_threshold",
        type=float,
        default=1.0,
        help="Std deviation threshold for large traces (default: 1.0).",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=2,
        help="Number of clusters for K-means (default: 2).",
    )

    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )
    args = parser.parse_args()

    p = dict()

    p["trace_files"] = []
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
            p["trace_files"] = [line.rstrip("\n") for line in sys.stdin]
        else:
            print("Nothing in stdin")
    else:
        p["pipe"] = False
        p["trace_files"] = [args.input]

    return args, p


def generate_unique_id():
    """Generates a unique identifier for a trace."""
    return str(uuid.uuid4())


def compute_radius_of_gyration(coords):
    """Computes the radius of gyration (Rg) for a given trace."""
    center_of_mass = np.mean(coords, axis=0)
    return np.sqrt(np.mean(np.sum((coords - center_of_mass) ** 2, axis=1)))


def split_large_traces(trace_table, std_threshold, num_clusters):
    """
    Identifies traces with large Rg and applies K-means clustering to split them.

    Parameters:
    ----------
    trace_table : ChromatinTraceTable
        Input chromatin trace table.
    std_threshold : float
        Number of standard deviations above mean Rg to classify as large.
    num_clusters : int
        Number of clusters for K-means.

    Returns:
    -------
    None (modifies trace_table in place)
    """
    trace_table_by_id = trace_table.data.group_by("Trace_ID")
    rg_values = [
        compute_radius_of_gyration(np.vstack((trace["x"], trace["y"], trace["z"])).T)
        for trace in trace_table_by_id.groups
    ]

    mean_rg, std_rg = np.mean(rg_values), np.std(rg_values)
    rg_threshold = mean_rg + std_threshold * std_rg

    print(
        f"$ Mean Rg: {mean_rg:.3f}, Std Rg: {std_rg:.3f}, Threshold: {rg_threshold:.3f}"
    )

    new_trace_table = trace_table.data.copy()
    num_splits = 0

    for sub_trace_table in trace_table_by_id.groups:
        original_trace_id = sub_trace_table["Trace_ID"][0]
        coords = np.vstack(
            (sub_trace_table["x"], sub_trace_table["y"], sub_trace_table["z"])
        ).T
        rg = compute_radius_of_gyration(coords)

        if rg > rg_threshold and len(coords) > num_clusters:
            print(
                f"$ Splitting trace {original_trace_id} (Rg={rg:.3f}) into {num_clusters} clusters."
            )
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)

            for cluster_label in np.unique(labels):
                new_trace_id = generate_unique_id()
                original_indices = np.where(
                    trace_table.data["Trace_ID"] == original_trace_id
                )[0]
                cluster_indices = original_indices[np.where(labels == cluster_label)[0]]
                new_trace_table["Trace_ID"][cluster_indices] = new_trace_id

            num_splits += 1

    print(f"$ Number of traces split: {num_splits}/{len(trace_table_by_id.groups)}")
    trace_table.data = new_trace_table


def main():
    """Main function to handle input, processing, and output."""
    args, p = parse_arguments()

    trace_files = p["trace_files"]
    if len(trace_files) > 0:
        print(
            "\n{} trace files to process= {}".format(
                len(trace_files), "\n".join(map(str, trace_files))
            )
        )

        # iterates over traces in folder
        for trace_file in trace_files:

            output_filename = (
                args.output
                if args.output
                else f"{os.path.splitext(trace_file)[0]}_split.ecsv"
            )

            trace_table = ChromatinTraceTable()
            trace_table.load(trace_file)

            print(
                f"Applying K-means clustering with {args.num_clusters} clusters on traces with Rg > mean + {args.std_threshold} * std_dev..."
            )
            split_large_traces(trace_table, args.std_threshold, args.num_clusters)

            trace_table.save(output_filename, trace_table.data)
            # print(f"Saved modified trace table: {output_filename}")

    else:
        print(
            "! Error: did not find any trace file to analyze. Please provide one using --input or --pipe."
        )


if __name__ == "__main__":
    main()
