#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 08:43:49 2022

@author: marcnol
- This script takes JSON file with folders where datasets are
stored.
- It searches for Trace files with the expected methods, loads them, and
- combines them into a single table that is outputted to the buildPWDmatrix folder.

$ trace_combinator.py

outputs

ChromatinTraceTable() object and output .ecsv formatted file with assembled trace tables.


"""

# =============================================================================
# IMPORTS
# =============================================================================q

import argparse
import glob
import json
import os
import select
import sys

from matrixOperations.chromatin_trace_table import ChromatinTraceTable

# =============================================================================
# FUNCTIONS
# =============================================================================q


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images")
    parser.add_argument(
        "-P",
        "--parameters",
        help="Provide name of parameter files. folders_to_load.json assumed as default",
    )
    parser.add_argument("-A", "--label", help="Add name of label (e.g. doc)")
    parser.add_argument(
        "-W", "--action", help="Select: [all], [labeled] or [unlabeled] cells plotted "
    )
    parser.add_argument(
        "--saveMatrix", help="Use to load matlab formatted data", action="store_true"
    )
    parser.add_argument("--ndims", help="Dimensions of trace")
    parser.add_argument(
        "--method", help="Method or mask ID used for tracing: KDtree, mask, DAPI"
    )
    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    p = {}

    args = parser.parse_args()
    if args.rootFolder:
        p["rootFolder"] = args.rootFolder
    else:
        p["rootFolder"] = "."

    if args.parameters:
        p["parametersFileName"] = args.parameters
    else:
        p["parametersFileName"] = "folders_to_load.json"

    if args.label:
        p["label"] = args.label
    else:
        p["label"] = "M"

    if args.action:
        p["action"] = args.action
    else:
        p["action"] = "all"

    if args.saveMatrix:
        p["saveMatrix"] = True
    else:
        p["saveMatrix"] = False

    if args.ndims:
        p["ndims"] = args.ndims
    else:
        p["ndims"] = 3

    if args.method:
        p["method"] = args.method
    else:
        p["method"] = "mask"

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
            print("Nothing in stdin!\n")
    else:
        p["pipe"] = False

    print("Input parameters\n" + "-" * 15)
    for item in p.keys():
        print("{}-->{}".format(item, p[item]))

    return p


def filter_trace(trace, label, action):
    rows_to_remove = []
    number_original_traces = len(trace.data)

    # print(f"label:{label}|action:{action}")
    if "all" not in action:  # otherwise it keeps all rows
        # finds rows to remove
        for index, trace_row in enumerate(trace.data):
            labels = list(trace_row["label"].split(","))
            if ("unlabeled" in action) and (label in labels):
                # removes labeled
                rows_to_remove.append(index)
            elif (
                ("labeled" in action)
                and ("unlabeled" not in action)
                and (label not in labels)
            ):
                # removes unlabeled
                rows_to_remove.append(index)

        # removes rows
        trace.data.remove_rows(rows_to_remove)

    print(
        f"> {number_original_traces-len(rows_to_remove)}/{number_original_traces} trace rows with <{label}> selected: ~{int(100*(number_original_traces-len(rows_to_remove))/number_original_traces)}%"
    )

    return trace


def appends_traces(traces, trace_files, label, action):
    new_trace = ChromatinTraceTable()

    # iterates over traces in folder
    for trace_file in trace_files:
        # reads new trace
        new_trace.load(trace_file)

        # filters rows based on label
        new_trace = filter_trace(new_trace, label, action)

        # adds it to existing trace collection

        traces.append(new_trace.data)
        traces.number_traces += 1

    return traces


def load_traces(
    folders=[],
    ndims=3,
    method="mask",
    label="none",
    action="all",
    trace_files=[],
):
    traces = ChromatinTraceTable()
    traces.initialize()
    traces.number_traces = 0

    if len(trace_files) < 1:
        # user provided a list of folders in folders_to_load.json
        for folder in folders:
            trace_files = [
                x
                for x in glob.glob(folder.rstrip(os.sep) + os.sep + "Trace*ecsv")
                if "uniqueBarcodes" not in x
            ]
            trace_files = [
                x for x in trace_files if (str(ndims) + "D" in x) and (method in x)
            ]

            print(
                "\n{} trace files to process= {}".format(
                    len(trace_files), "\n--> ".join(map(str, trace_files))
                )
            )

            if len(trace_files) > 0:
                traces = appends_traces(traces, trace_files, label, action)
    else:
        # user provided a list of files to concatenate
        if len(trace_files) > 0:
            traces = appends_traces(traces, trace_files, label, action)

    print(
        f"Read and accumulated {traces.number_traces} trace files with ndims = {ndims} and method = {method}"
    )

    return traces


def run(p):
    # [ Lists and loads datasets from different embryos]
    input_parameters = p["rootFolder"] + os.sep + p["parametersFileName"]
    print("\n" + "-" * 80)

    if not p["pipe"]:
        if os.path.exists(input_parameters):
            with open(input_parameters, encoding="utf-8") as json_file:
                data_dict = json.load(json_file)
            print(
                "Loaded JSON file with {} datasets from {}\n".format(
                    len(data_dict), input_parameters
                )
            )
        else:
            print("File not found: {}".format(input_parameters))
            sys.exit()

    # [ creates output folder]
    p["outputFolder"] = p["rootFolder"] + os.sep + "combined_traces"
    if not os.path.exists(p["outputFolder"]):
        os.mkdir(p["outputFolder"])
        print("Folder created: {}".format(p["outputFolder"]))

    if not p["pipe"]:
        # [loops over lists of datafolders]
        dataset_names = list(data_dict.keys())
        for dataset in dataset_names:
            traces = load_traces(
                folders=data_dict[dataset]["Folders"],
                ndims=p["ndims"],
                method=p["method"],
                label=p["label"],
                action=p["action"],
            )
    else:
        traces = load_traces(
            ndims=p["ndims"],
            method=p["method"],
            label=p["label"],
            action=p["action"],
            trace_files=p["trace_files"],
        )

    outputfile = (
        p["outputFolder"]
        + os.sep
        + "Trace_combined_"
        + str(p["ndims"])
        + "D_method:"
        + p["method"]
        + "_label:"
        + p["label"]
        + "_action:"
        + p["action"]
        + ".ecsv"
    )

    traces.save(
        outputfile,
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
    if p["pipe"] and len(p["trace_files"]) < 1:
        print("\nNothing to process...\n")
    else:
        run(p)


if __name__ == "__main__":
    main()
