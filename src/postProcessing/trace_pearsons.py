#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# trace_pearsons.py

## Description
A script for comparing multiple chromatin trace tables by computing pairwise distances
between barcode combinations and quantifying similarity via Pearson correlation.

This tool is useful for analyzing the structural similarity between different chromatin
trace datasets, helping to identify patterns and relationships in chromatin organization
across multiple samples or conditions.

## Usage
```bash
$ ls *ecsv | python trace_pearsons.py [options]
$ find . -name "*.ecsv" | python trace_pearsons.py [options]
```

## Arguments
- `--output` - Output filename for the correlation matrix plot (default: trace_correlation_matrix.png)

## Output
- A Pearson correlation matrix plot comparing all input trace tables
- Terminal output showing the numerical correlation values

## Notes
- Input files must be in ECSV format compatible with ChromatinTraceTable
- The script identifies unique parts of filenames to create readable labels in the plot
- Correlation is calculated based on the spatial distances between barcode pairs
"""

import argparse
import itertools
import os
import select
import sys
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from matrixOperations.chromatin_trace_table import ChromatinTraceTable


def parse_arguments():
    """
    Parse command line arguments for the script.

    Returns:
    -------
    tuple
        (args, trace_files) where args is the parsed arguments and
        trace_files is a list of input files from stdin
    """
    parser = argparse.ArgumentParser(
        description="Compare chromatin trace tables using Pearson correlation."
    )

    parser.add_argument(
        "--output",
        default="trace_correlation_matrix.png",
        help="Output filename for the correlation matrix plot (default: trace_correlation_matrix.png)",
    )

    trace_files = []
    if select.select([sys.stdin], [], [], 0.0)[0]:
        trace_files = [line.rstrip("\n") for line in sys.stdin]
    else:
        print(
            "Nothing in stdin! Please provide list of tracefiles as in:\n$ ls *ecsv | python trace_pearsons.py"
        )

    args = parser.parse_args()

    return args, trace_files


def find_unique_substrings(filenames):
    """
    Find unique substrings in each filename that distinguish them from all others.

    This function identifies the minimal substring that makes each filename unique
    within the given set, without relying on predefined patterns.

    Parameters:
    ----------
    filenames : list
        List of filenames to analyze

    Returns:
    -------
    dict
        Dictionary mapping each filename to its unique identifying substring
    """

    # Create a list of tuples with each filename and its character-by-character comparison
    file_chars = []
    for filename in filenames:
        file_chars.append((filename, list(filename)))

    # Dictionary to store unique identifiers
    unique_identifiers = {}

    for i, (filename, chars) in enumerate(file_chars):
        # Compare with all other filenames
        differing_positions = []

        for j, (other_filename, other_chars) in enumerate(file_chars):
            if i == j:  # Skip self-comparison
                continue

            # Find differing positions
            min_len = min(len(chars), len(other_chars))
            for pos in range(min_len):
                if chars[pos] != other_chars[pos]:
                    differing_positions.append(pos)

            # Handle case where one filename is a prefix of another
            if len(chars) != len(other_chars):
                for pos in range(min_len, max(len(chars), len(other_chars))):
                    differing_positions.append(pos)

        # Get all unique positions where this file differs from others
        unique_positions = sorted(set(differing_positions))

        if not unique_positions:
            unique_identifiers[filename] = (
                ""  # No unique part (should not happen with different filenames)
            )
            continue

        # Find consecutive ranges of differing positions
        ranges = []
        current_range = [unique_positions[0]]

        for pos in unique_positions[1:]:
            if pos == current_range[-1] + 1:
                current_range.append(pos)
            else:
                ranges.append(current_range)
                current_range = [pos]

        ranges.append(current_range)  # Add the last range

        # Find the most significant range (typically the longest or most meaningful)
        # For simplicity, we'll use the longest range
        longest_range = max(ranges, key=len)

        # Extract the unique substring
        start = longest_range[0]
        end = longest_range[-1] + 1
        unique_substring = filename[start:end]

        # Clean up the substring - remove partial words or patterns
        # This attempts to find natural word boundaries around the unique part
        expanded_start = start
        expanded_end = end

        # Expand backward to include a word boundary or common delimiter
        while expanded_start > 0 and filename[expanded_start - 1] not in [
            " ",
            "_",
            "-",
            ".",
        ]:
            expanded_start -= 1

        # Expand forward to include a word boundary or common delimiter
        while expanded_end < len(filename) and filename[expanded_end] not in [
            " ",
            "_",
            "-",
            ".",
        ]:
            expanded_end += 1

        # Use the expanded substring if it's not too much longer
        if (expanded_end - expanded_start) <= 2 * (end - start):
            unique_substring = filename[expanded_start:expanded_end]

        # remove extension
        unique_substring = unique_substring.split(".")[0]

        unique_identifiers[filename] = unique_substring.strip()

    return unique_identifiers


def accumulate_distances(trace_data):
    """
    Calculate pairwise distances between barcodes across all traces.

    Parameters:
    ----------
    trace_data : ChromatinTraceTable
        Table containing trace data with barcode positions

    Returns:
    -------
    dict
        Dictionary mapping barcode pairs to their median distances
    """
    distances = defaultdict(list)
    trace_groups = trace_data.group_by("Trace_ID").groups

    for trace in trace_groups:
        barcode_positions = {
            row["Barcode #"]: np.array([row["x"], row["y"], row["z"]])
            for row in trace
            if not np.isnan(row["x"])
        }
        for bc1, bc2 in combinations(barcode_positions, 2):
            p1 = barcode_positions[bc1]
            p2 = barcode_positions[bc2]
            dist = np.linalg.norm(p1 - p2)
            key = tuple(sorted((bc1, bc2)))
            distances[key].append(dist)

    return {key: np.median(vals) for key, vals in distances.items()}


def compare_distance_maps(distance_maps):
    """
    Calculate Pearson correlation between distance maps from different trace files.

    Parameters:
    ----------
    distance_maps : dict
        Dictionary mapping filenames to their distance maps

    Returns:
    -------
    tuple
        (files, corr_matrix) where files is a list of filenames and
        corr_matrix is the correlation matrix
    """
    files = list(distance_maps.keys())
    all_keys = set(
        itertools.chain.from_iterable([dm.keys() for dm in distance_maps.values()])
    )

    # Create vectors for each file over all keys
    vectors = {}
    for fname in files:
        vec = []
        for key in sorted(all_keys):
            vec.append(distance_maps[fname].get(key, np.nan))
        vectors[fname] = np.array(vec)

    # Build correlation matrix
    n = len(files)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = vectors[files[i]]
            y = vectors[files[j]]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(mask) > 1:
                corr, _ = pearsonr(x[mask], y[mask])
            else:
                corr = np.nan
            corr_matrix[i, j] = corr

    return files, corr_matrix


def plot_correlation_matrix(
    files, matrix, output_filename="trace_correlation_matrix.png"
):
    """
    Plot a correlation matrix between files with unique identifiers as labels.

    Parameters:
    ----------
    files : list
        List of filenames to use
    matrix : numpy.ndarray
        Square correlation matrix of the files
    output_filename : str
        Name of the output file (default: trace_correlation_matrix.png)

    Returns:
    -------
    None
        Saves the plot as a PNG file
    """
    # Get unique identifiers for each file
    unique_identifiers = find_unique_substrings(files)

    # Create labels for the plot
    labels = [os.path.basename(unique_identifiers[f]) for f in files]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the matrix with dynamic color range
    vmin = np.min(matrix)
    vmax = np.max(matrix)
    im = ax.imshow(matrix, cmap="RdBu", interpolation="nearest", vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson Correlation", fontsize=12)

    # Set tick labels
    ax.set_xticks(range(len(files)))
    ax.set_yticks(range(len(files)))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add axis labels
    ax.set_xlabel("Files", fontsize=14)
    ax.set_ylabel("Files", fontsize=14)

    # Add title
    ax.set_title("Trace Table Similarity Matrix", fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"Saved correlation matrix as {output_filename}")


def main():
    """
    Main function that executes the trace comparison workflow.
    """
    args, trace_files = parse_arguments()

    if not trace_files:
        return

    print(f"Analyzing {len(trace_files)} trace files...")

    # Calculate distance maps for each file
    distance_maps = {}
    for fpath in trace_files:
        print(f"Processing {os.path.basename(fpath)}")
        trace = ChromatinTraceTable()
        trace.load(fpath)
        distance_maps[fpath] = accumulate_distances(trace.data)

    # Compare distance maps and generate correlation matrix
    files, corr_matrix = compare_distance_maps(distance_maps)

    print("\nPearson Correlation Matrix:")
    print(corr_matrix)

    # Plot and save the correlation matrix
    plot_correlation_matrix(files, corr_matrix, output_filename=args.output)


if __name__ == "__main__":
    main()
