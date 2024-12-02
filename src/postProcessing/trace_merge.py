#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 15 2023

@author: marcnol

Simpler version of trace_combinator.

This just takes a list of trace files and merges them together

$ ls Trace*.ecsv | trace_merge.py

outputs

ChromatinTraceTable() object and output .ecsv formatted file with assembled trace tables.


"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import os
import select
import sys

from matrixOperations.chromatin_trace_table import ChromatinTraceTable

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_file", help="Output File name. Default = merged_traces.ecsv"
    )
    parser.add_argument("-O", "--output_folder", help="Output File name. Default = ./")
    p = {}

    args = parser.parse_args()
    if args.output_folder:
        p["outputFolder"] = args.output_folder
    else:
        p["outputFolder"] = "."

    if args.output_file:
        p["output_file"] = args.output_file
    else:
        p["output_file"] = "merged_traces.ecsv"

    p["trace_files"] = []
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
        print("Nothing in stdin!\n")

    print("Input parameters\n" + "-" * 15)
    for item in p.keys():
        print("{}-->{}".format(item, p[item]))

    return p


def appends_traces(traces, trace_files):
    new_trace = ChromatinTraceTable()

    # iterates over traces in folder
    for trace_file in trace_files:
        # reads new trace
        new_trace.load(trace_file)

        # adds it to existing trace collection
        traces.append(new_trace.data)
        traces.number_traces += 1

        print(f" $ appended trace file with {len(new_trace.data)} traces")

    print(f" $ Merged trace file will contain {len(traces.data)} traces")

    return traces


def load_traces(trace_files=[]):
    traces = ChromatinTraceTable()
    traces.initialize()
    traces.number_traces = 0

    if len(trace_files) > 1:
        # user provided a list of files to concatenate
        traces = appends_traces(traces, trace_files)

    print(f"Read and accumulated {traces.number_traces} trace files")

    return traces


def run(p):
    print("\n" + "-" * 80)

    # [ creates output folder]
    if not os.path.exists(p["outputFolder"]):
        os.mkdir(p["outputFolder"])
        print("Folder created: {}".format(p["outputFolder"]))

    # loads and merges traces
    traces = load_traces(trace_files=p["trace_files"])

    # saves merged trace table
    output_file = p["output_file"]

    traces.save(
        output_file,
        traces.data,
        comments="appended_trace_files=" + str(traces.number_traces),
    )

    print("Finished execution")


# =============================================================================
# MAIN
# =============================================================================


def main():
    # [parsing arguments]
    p = parse_arguments()

    print("trace_files{}".format(len(p["trace_files"])))
    if len(p["trace_files"]) < 1:
        print("\nNothing to process...\n")
    else:
        run(p)


if __name__ == "__main__":
    main()
