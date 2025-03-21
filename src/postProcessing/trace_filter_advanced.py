#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:

$ trace_filter.py --input Trace.ecsv --N_barcodes 3 --fraction_missing_barcodes -0.5 --overlapping_threshold 0.03

will analyze 'Trace.ecsv' and remove traces with:


- less than 3 barcodes
- fraction of missing barcodes < 0.5
- barcodes closer than 0.03 um will be merged.


outputs:

.ecsv trace table file with the '_filtered' tag appended.


"""

import argparse
import os
import select
import sys
import uuid

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from tqdm import tqdm

from core.data_manager import create_folder

# matplotlib.use('TkAgg')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--rootFolder", help="Folder with images")
    parser.add_argument(
        "-O", "--output", help="Tag to add to the output file. Default = filtered"
    )
    parser.add_argument(
        "--fraction_missing_barcodes",
        help="fraction of missing barcodes. Default = 0.5",
    )
    parser.add_argument("--input", help="Name of input trace file.")
    parser.add_argument(
        "--overlapping_threshold", help="overlapping threshold. Default = 0.030"
    )
    parser.add_argument("--N_barcodes", help="minimum_number_barcodes. Default = 2")

    parser.add_argument(
        "--pipe", help="inputs Trace file list from stdin (pipe)", action="store_true"
    )

    p = {}

    args = parser.parse_args()
    if args.output:
        p["output"] = args.output
    else:
        p["output"] = "filtered"

    args = parser.parse_args()
    if args.rootFolder:
        p["rootFolder"] = args.rootFolder
    else:
        p["rootFolder"] = "."

    if args.input:
        p["input"] = args.input
    else:
        p["input"] = None

    if args.fraction_missing_barcodes:
        p["fraction_missing_barcodes"] = args.fraction_missing_barcodes
    else:
        p["fraction_missing_barcodes"] = 0.5

    if args.overlapping_threshold:
        p["overlapping_threshold"] = args.overlapping_threshold
    else:
        p["overlapping_threshold"] = 0.030

    if args.overlapping_threshold:
        p["overlapping_threshold"] = args.overlapping_threshold
    else:
        p["overlapping_threshold"] = 0.030

    if args.N_barcodes:
        p["N_barcodes"] = int(args.N_barcodes)
    else:
        p["N_barcodes"] = 2

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
        p["trace_files"] = [p["input"]]

    return p


def plot_repeated_barcodes(trace_data):
    """Plot a 3d graph with all the localizations. For the repeated barcodes, the localizations are plotted with a
    specific legend.

    @param (pandas dataframe) input data with all the traces & detections
    """
    unique_bc = trace_data["Barcode #"].drop_duplicates()
    unique_bc = sorted(unique_bc.values.tolist())
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        trace_data["x"], trace_data["y"], trace_data["z"], facecolor="0.5", alpha=0.5
    )
    for bc in unique_bc:
        subset = trace_data[trace_data["Barcode #"] == bc]
        if len(subset) > 1:
            ax.scatter(subset["x"], subset["y"], subset["z"])

            pos = np.zeros((len(subset), 3))
            pos[:, 0] = subset["x"]
            pos[:, 1] = subset["y"]
            pos[:, 2] = subset["z"]
            for n in range(len(subset)):
                ax.text(
                    subset.iloc[n]["x"],
                    subset.iloc[n]["y"],
                    subset.iloc[n]["z"],
                    bc,
                    [1, 1, 0],
                )
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    plt.show()


class FilterTraces:
    def __init__(self, data_folder, data_file, dest_folder, threshold=0, verbose=False):
        self.data_folder: str = data_folder
        self.data_file: str = data_file
        self.dest_folder: str = dest_folder
        self.overlapping_threshold: float = threshold  # in µm
        self.verbose: bool = verbose
        self.data = None
        self.p95: float = 0
        self.p99: float = 0
        self.clustered_data = None
        self.trace_filename: str = ""

        # load the traces and analyze the file
        self.open_him_traces()
        self.n_bin = len(
            self.data["Barcode #"].drop_duplicates()
        )  # total number of unique barcodes found
        self.bc = self.data["Barcode #"].drop_duplicates()
        self.n_traces_total = len(
            self.data["Trace_ID"].drop_duplicates()
        )  # number of unique trace id
        self.trace_id = self.data[
            "Trace_ID"
        ].drop_duplicates()  # list of all the unique trace id
        self.unique_labels = self.data["label"].drop_duplicates()

    def open_him_traces(self):
        """Open HiM trace file and convert it to panda dataframe."""

        # define the path to the trace file
        # check a file was found, else exit the method

        file_path = self.data_folder + os.sep + self.data_file
        if len(file_path) == 0:
            print("No trace file was found. The loading of the traces is aborted")
            return
        else:
            # load the trace files and eventually concatenate them together
            dataframe = []
            try:
                data = ascii.read(file_path, format="ecsv", delimiter=" ")
                data = data.to_pandas()
                dataframe.append(data)
            except Exception as err:
                print(f"Unexpected {err} for file {file_path}")

            self.data = pd.concat(dataframe)

    def hard_filtering(self):
        """Filtering the originally loaded traces by removing all the trace with at least one duplicated barcode.

        @return: (panda dataframe) filtered traces
        """
        new_dataframe = self.data.copy()

        for id in tqdm(self.trace_id):
            single_trace = self.data.loc[self.data["Trace_ID"] == id]
            n_barcodes = len(single_trace["Barcode #"])
            n_unique_barcodes = len(single_trace["Barcode #"].drop_duplicates())
            if (n_barcodes != n_unique_barcodes) or (n_unique_barcodes < 2):
                new_dataframe.drop(
                    new_dataframe[new_dataframe["Trace_ID"] == id].index, inplace=True
                )

        return new_dataframe

    @staticmethod
    def select_traces_wo_duplicates(data, N_barcodes=2):
        """Analyze the trace dataframe and select only the traces with no duplicates and containing at least 2
        barcodes.

        @type data: (dataframe) input trace on which the analysis is performed
        @return: (list) list of all the trace_ID selected
        """
        id_wo_duplicates = []
        trace_id = data["Trace_ID"].drop_duplicates()

        for id in trace_id:
            single_trace = data.loc[data["Trace_ID"] == id]
            n_barcodes = len(single_trace["Barcode #"])
            n_unique_barcodes = len(single_trace["Barcode #"].drop_duplicates())
            if (n_barcodes == n_unique_barcodes) and (n_unique_barcodes >= N_barcodes):
                id_wo_duplicates.append(id)

        return id_wo_duplicates

    def calculate_pwd_threshold(self, trace_id, verbose=False, save=False):
        """For all the traces, calculated the pairwise distance between all the detections. From the distribution,
        calculate the 95% and 99% quantiles.

        @param trace_id: (list) list of all the ID of the traces without duplicated barcodes
        @param verbose: (bool) indicate whether the distribution should be plotted
        @param save: (bool) indicate whether the plot should be saved instead of being displayed in a popup window
        @return: p95 and p99 (float) for the values of the 95% and 99% quantiles
        """
        pwd_distribution = []
        print(f"$ Will process {len(trace_id)} traces")
        for id in trace_id:
            trace = self.data[self.data["Trace_ID"] == id]
            pos = trace[["x", "y", "z"]].to_numpy()
            if pos.shape[0] > 1:
                distance = pairwise_distances(pos)
                distance_flatten = self.him_map_2d_to_1d(distance)
                pwd_distribution.append(distance_flatten)

        pwd_distribution = [item for sublist in pwd_distribution for item in sublist]
        pwd_distribution = np.asarray(pwd_distribution, dtype=object)
        n_bin = (
            (np.max(pwd_distribution) - np.min(pwd_distribution)) * 1000 / 10
        )  # in order to get ~10nm / bin
        n_bin = np.round(n_bin).astype(np.int16)

        med = np.around(np.median(pwd_distribution), decimals=2)
        self.p95 = np.around(np.quantile(pwd_distribution, 0.95), decimals=2)
        self.p99 = np.around(np.quantile(pwd_distribution, 0.99), decimals=2)

        if verbose:
            plt.figure()
            plt.hist(np.transpose(pwd_distribution), bins=n_bin)
            plt.xlabel("pairwise distance (µm)")
            plt.ylabel("number of occurrences")
            plt.title(
                f"Median={med}µm - quantile 95%={self.p95}µm - quantile 99%={self.p99}µm"
            )
            if save:
                fig_path = os.path.join(self.dest_folder, "pairwise_distance_stat.png")
                plt.savefig(fig_path, dpi=200, format="png")
            else:
                plt.show()

        return self.p95, self.p99

    def trace_statistics(self, save=True, tag=""):
        """plot the statistics for the selected traces. Two plots are displayed :
        1- for each barcode, indicate the number of detected spots as well as the proportion of duplicated barcodes
        2- the detection efficiency, that is the number of traces with a given proportion of detected barcodes. Again,
        the proportion of traces with duplicated barcodes is indicated

        @param save: (bool) indicate if the figure should be saved instead of displayed
        @param tag: (str) string to add to the image name
        """
        efficacy_wo_duplicates = []
        efficacy_w_duplicates = []
        barcode_detection_single = []
        barcode_detection_duplicated = []
        drop_out = 0
        for id in tqdm(self.trace_id):
            # select all the detections belonging to the trace with id and calculate the detection efficiency for this
            # trace as well as the number of duplicated barcodes. If the trace contains a single detection, it is
            # counted as a dropout.
            single_trace = self.data.loc[self.data["Trace_ID"] == id]
            n_barcodes = len(single_trace["Barcode #"])
            n_unique_barcodes = len(single_trace["Barcode #"].drop_duplicates())
            efficacy = np.around(n_unique_barcodes * 100 / self.n_bin, decimals=1)
            if (n_barcodes == n_unique_barcodes) and (n_unique_barcodes > 1):
                efficacy_wo_duplicates.append(efficacy)
            elif (n_barcodes != n_unique_barcodes) and (n_unique_barcodes > 1):
                efficacy_w_duplicates.append(efficacy)
            else:
                drop_out += 1

            # each barcode composing the trace is also counted. If a single detection is detected, it is appended to
            # barcode_detection_single, else to barcode_detection_duplicated.
            for bc in single_trace["Barcode #"].drop_duplicates().to_list():
                if len(single_trace[single_trace["Barcode #"] == bc]) == 1:
                    barcode_detection_single.append(bc)
                else:
                    barcode_detection_duplicated.append(bc)

        # plot the two graphs
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        fig1.set_figheight(10)
        fig1.set_figwidth(20)
        bc_unique = list(set(barcode_detection_single + barcode_detection_duplicated))
        barcode_detection_single_stat = [
            barcode_detection_single.count(bc) for bc in bc_unique
        ]
        barcode_detection_duplicated_stat = [
            barcode_detection_duplicated.count(bc) for bc in bc_unique
        ]
        bc_unique = [str(bc) for bc in bc_unique]

        ax1.bar(
            bc_unique,
            barcode_detection_single_stat,
            width=0.8,
            label="Single detection",
        )
        ax1.bar(
            bc_unique,
            barcode_detection_duplicated_stat,
            width=0.8,
            bottom=barcode_detection_single_stat,
            label="Duplicated detections",
        )
        ax1.set_xlabel("Barcode names")
        ax1.set_ylabel("Number of detected spots")
        ax1.legend()

        efficacy_unique = list(set(efficacy_wo_duplicates + efficacy_w_duplicates))
        efficacy_wo_duplicates_stat = [
            efficacy_wo_duplicates.count(eff) for eff in efficacy_unique
        ]
        efficacy_w_duplicates_stat = [
            efficacy_w_duplicates.count(eff) for eff in efficacy_unique
        ]

        ax2.bar(
            efficacy_unique,
            efficacy_wo_duplicates_stat,
            width=3,
            label="Without duplicates",
        )
        ax2.bar(
            efficacy_unique,
            efficacy_w_duplicates_stat,
            width=3,
            bottom=efficacy_wo_duplicates_stat,
            label="With duplicates",
        )
        ax2.set_xlabel("Detection efficiency (%)")
        ax2.set_ylabel("Number of traces")
        ax2.set_title(f"N_traces = {len(self.trace_id)} - dropout = {drop_out}")
        ax2.legend()

        if save:
            fig_path = os.path.join(self.dest_folder, f"trace_stat_{tag}.png")
            plt.savefig(fig_path, dpi=200, format="png")
        else:
            plt.show()

    def filter_traces(self, verbose=False):
        """All the traces are analyzed based on their ID. Using a clustering algorithm and the threshold calculated
        based on the pwd distribution, each trace is redefined as a list of spot_ID and kept in "updated_spot_id". That
        way, traces composed of multiple duplicated barcodes can now be separated into multiple sub-traces, each
        represented as a single list of spot_ID. All the isolated detections (not associated to a trace) are discarded.
        Same for the traces presenting less than 20% of the available barcodes.

        @param verbose: (bool) indicate whether the plot should be displayed
        @return: filtered_data (pandas dataframe) output a new dataframe with the updated traces associated to a new
        unique ID
        """
        updated_spot_id = []
        discarded_spot_id = []

        # for each single trace, launch the clusterization algorithm
        # ---------------------------------------------------------
        print("\n$ Performing clusterization ...")
        for trace_id in tqdm(self.trace_id):
            single_trace = self.data.loc[self.data["Trace_ID"] == trace_id]
            kept_id, out_id = self.clustering(
                single_trace, self.p95, self.p99, verbose=False
            )
            for ids in kept_id:
                updated_spot_id.append(ids)
            if out_id:
                discarded_spot_id.append(out_id)

        # reformat the dataframe
        # ----------------------
        discarded_spot_id = [item for sublist in discarded_spot_id for item in sublist]
        filtered_data = self.reformat_dataframe(
            self.data, updated_spot_id, discarded_spot_id
        )

        # plot the distribution of discarded barcodes
        # -------------------------------------------
        if verbose:
            bc_list = sorted(self.bc.to_list())
            discarded_bc = self.data.loc[
                self.data["Spot_ID"].isin(discarded_spot_id), "Barcode #"
            ].to_list()
            bc_stat = [discarded_bc.count(bc) for bc in bc_list]

            labels = [str(bc) for bc in bc_list]
            bc_stat = np.array(bc_stat)
            bc_stat = np.around(bc_stat * 100 / np.sum(bc_stat), decimals=1)

            fig1, ax1 = plt.subplots()
            ax1.pie(bc_stat, labels=labels, autopct="%1.1f%%", startangle=90)
            ax1.axis(
                "equal"
            )  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.show()

        # return the filtered traces
        # --------------------------

        return filtered_data

    @staticmethod
    def clustering(trace_data, radius_min, radius_max, verbose=False):
        """For each single trace, a KDTree is first calculated based on the 3d localizations. Using the lower-bound
        threshold, a "query-radius" is launched and the neighbors associated to each localization are found.
        An iterative process is launched in order to reconstruct the different clusters aggregated in the initial trace.

        @param trace_data: (pandas dataframe) data associated to a single trace defined by a unique trace_ID
        @param radius_min: (float) lower-bound threshold for the pwd between two barcodes (seeding of the cluster)
        @param radius_max: (float) higher-bound threshold for the pwd between two barcodes (maximum distance allowed)
        @param verbose: (bool) indicate whether the plot should be displayed
        @return: kept_spot_id (list) contains lists of spot_ID. Each list is a trace reconstructed by the clustering algorithm out_spot_id (list) contains all the spot_ID of the isolated detections
        """
        # load the detections of the selected trace
        # -----------------------------------------
        pos = trace_data[["x", "y", "z"]].to_numpy()
        spot_id = trace_data["Spot_ID"].to_list()

        # perform a KDTree search on the 3d positions of the input trace. For each position, defines a list of the
        # closest neighbors based on the value of radius. Initialize the cluster list by using the localization with the
        # highest number of neighbors.
        # -------------------
        tree = KDTree(pos, metric="euclidean")
        neighbors = tree.query_radius(pos, r=radius_min, return_distance=False)
        neighbors = sorted(
            neighbors, key=lambda a: len(a), reverse=True
        )  # sort neighbors based on number of detections
        clusters = [
            neighbors[0].tolist()
        ]  # initialize the first cluster as the largest one

        # for each localization, compare the list of closest neighbors to the clusters that have already been defined.
        # If there is at least one localization in common with the existing clusters, the new localizations are added.
        # Else a new cluster is defined.
        # ----------------------
        for loc in neighbors:
            new_cluster = True
            for n, cluster in enumerate(clusters):
                intersect = set(cluster).intersection(
                    loc.tolist()
                )  # check common localizations between loc and cluster
                if len(intersect) > 0:
                    new_cluster = False
                    new_pos = set(loc.tolist()) - set(cluster)
                    clusters[n] = cluster + list(new_pos)

            if new_cluster:
                clusters.append(loc.tolist())

        # analyze the clusters and keep only the ones containing at least 2 barcodes
        # --------------------------------------------------------------------------
        n_cluster = 0
        assigned_pos = []
        left_out_pos = []
        for cluster in clusters:
            if len(cluster) > 1:
                n_cluster += 1
                assigned_pos.append([[x, n_cluster] for x in cluster])
            else:
                left_out_pos.append(cluster)

        assigned_pos = [item for sublist in assigned_pos for item in sublist]
        assigned_pos = np.array(assigned_pos)
        left_out_pos = [item for sublist in left_out_pos for item in sublist]

        # check whether the left out positions could belong to a cluster based on the maximum distance. If the same
        # point could be assigned to more than one cluster, it is left-out. If selected, the outsiders are added to a
        # new list (to avoid expanding the cluster while checking if outsiders could be stitched)
        # --------------------------------------------------------------------------
        kept_spot_id = []
        if n_cluster > 0:
            new_assigned_pos = np.copy(assigned_pos)
            for outsider in left_out_pos:
                d = pairwise_distances(
                    pos[outsider, :].reshape(1, -1), pos[assigned_pos[:, 0], :]
                )
                close_cluster = np.unique(assigned_pos[d[0, :] < radius_max, 1])

                if len(close_cluster) == 1:
                    np.append(
                        new_assigned_pos,
                        np.array([outsider, close_cluster], dtype=object),
                    )
                    left_out_pos.remove(outsider)

            # create a final list where all detections ID found within a single cluster are grouped together
            # ----------------------------------------------------------------------------------------------
            for n in range(n_cluster):
                index = new_assigned_pos[new_assigned_pos[:, 1] == n + 1, 0]
                detection_ids = [spot_id[int(x)] for x in index]
                kept_spot_id.append(detection_ids)

        # create a final list where the id of all the discarded detections are saved
        # --------------------------------------------------------------------------
        out_spot_id = [spot_id[int(x)] for x in left_out_pos]

        # plot the cluster (if Verbose option is True)
        # --------------------------------------------
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            for n in range(n_cluster):
                cluster = new_assigned_pos[new_assigned_pos[:, 1] == n + 1, 0]
                ax.scatter(pos[cluster, 0], pos[cluster, 1], pos[cluster, 2])
            for outsider in left_out_pos:
                ax.scatter(
                    pos[outsider, 0],
                    pos[outsider, 1],
                    pos[outsider, 2],
                    marker="*",
                    facecolor="0.5",
                )

            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_zlabel("z (µm)")
            plt.show()

        return kept_spot_id, out_spot_id

    @staticmethod
    def reformat_dataframe(dataframe, in_spot_id, out_spot_id):
        """Based on the list of spot_ID selected for the trace, reformat the dataframe by reassigning to all the new
        traces a unique ID. All the detections not associated to a trace are removed from the dataframe.

        @param dataframe: (pandas dataframe) input data with all the traces & detections
        @param in_spot_id: (list) contains lists of spot_ID, each defining a single unique trace
        @param out_spot_id: (list) contains the spot_ID of all the discarded detections
        @return: new_dataframe (pandas dataframe) with the new traces and their unique ID
        """
        new_dataframe = dataframe.copy()

        # for each cluster/trace, reassigned a unique trace_id to all the detections composing it
        # ---------------------------------------------------------------------------------------
        for spot_id in tqdm(in_spot_id):
            new_trace_id = str(uuid.uuid4())
            new_dataframe.loc[new_dataframe["Spot_ID"].isin(spot_id), "Trace_ID"] = (
                new_trace_id
            )

        # remove all the detections that were left-out
        # --------------------------------------------
        new_dataframe.drop(
            new_dataframe[new_dataframe["Spot_ID"].isin(out_spot_id)].index,
            inplace=True,
        )

        return new_dataframe

    def detect_overlapping_barcodes(self, trace_data, verbose=False, save=True):
        """Detect barcodes that are duplicated within the same trace. If the distance between two barcodes is lower
        than a specific threshold d_min (overlapping_threshold), they are replaced by their average localization.

        @param trace_data: (pandas dataframe) input data with all the traces & detections
        @param verbose: (bool) indicate whether the plot should be displayed
        @return: new_trace_data (pandas dataframe) after removing the duplicated barcodes
        """
        all_trace_id = trace_data["Trace_ID"].drop_duplicates()
        d_min = (
            self.overlapping_threshold
        )  # distance threshold below which the two detections are replaced
        spot_id_to_keep = []
        spot_id_to_remove = []
        pwd_repeated_bc = []

        # In a first step, the trace is analyzed by sequentially selecting each barcode and checking that all
        # localizations are above the threshold
        for trace_id in tqdm(all_trace_id):
            single_trace = trace_data.loc[trace_data["Trace_ID"] == trace_id]
            trace_bc = single_trace["Barcode #"].tolist()
            unique_trace_bc = single_trace["Barcode #"].drop_duplicates().tolist()

            if len(unique_trace_bc) != len(trace_bc):
                repeated_bc = [bc for bc in unique_trace_bc if trace_bc.count(bc) > 1]
            else:
                continue

            for bc in repeated_bc:
                subset = single_trace.loc[single_trace["Barcode #"] == bc]
                pos = np.zeros((len(subset), 3))
                pos[:, 0] = subset["x"]
                pos[:, 1] = subset["y"]
                pos[:, 2] = subset["z"]

                # calculate the pairwise distances and remove the duplicated values (first half of the matrix)
                pwd = pairwise_distances(pos)
                for x in range(pwd.shape[0] - 1):
                    pwd[x, x + 1 : pwd.shape[0]] = 0

                # detect the pairs of barcodes that are below the threshold d_min and keep their id (Spot_ID)
                index = np.where((pwd < d_min) & (pwd > 0))
                n_overlap = len(index[0])
                if n_overlap > 0:
                    for idx in range(n_overlap):
                        spot_id_to_keep.append(subset.iloc[index[0][idx]]["Spot_ID"])
                        spot_id_to_remove.append(subset.iloc[index[1][idx]]["Spot_ID"])

                # save the pwd values in order to plot the distribution
                pwd = np.reshape(pwd, (pwd.shape[0] * pwd.shape[1]))
                pwd_repeated_bc.append([d for d in pwd.tolist() if d > 0])

        # plot the distribution of the pwd distance
        pwd_repeated_bc = [item for sublist in pwd_repeated_bc for item in sublist]
        if verbose:
            plt.figure()
            plt.hist(np.transpose(pwd_repeated_bc), bins=100)
            plt.xlabel("pairwise distance (µm)")
            plt.ylabel("number of occurrences")
            plt.title("distribution of pwd for the repeated barcodes")
            if save:
                fig_path = os.path.join(self.dest_folder, "duplicated_bc_pwd_stat.png")
                plt.savefig(fig_path, dpi=200, format="png")
            else:
                plt.show()

        # In a second step, a copy of the dataframe is performed (to avoid warnings due to chained indexing issue with
        # pandas) and replace the localizations of the selected duplicated barcodes by the average position.

        new_trace_data = trace_data.copy()

        for n, id_to_keep in enumerate(spot_id_to_keep):
            id_to_remove = spot_id_to_remove[n]
            x = (
                trace_data.loc[trace_data["Spot_ID"] == id_to_keep, "x"].iloc[0]
                + trace_data.loc[trace_data["Spot_ID"] == id_to_remove, "x"].iloc[0]
            ) / 2
            y = (
                trace_data.loc[trace_data["Spot_ID"] == id_to_keep, "y"].iloc[0]
                + trace_data.loc[trace_data["Spot_ID"] == id_to_remove, "y"].iloc[0]
            ) / 2
            z = (
                trace_data.loc[trace_data["Spot_ID"] == id_to_keep, "z"].iloc[0]
                + trace_data.loc[trace_data["Spot_ID"] == id_to_remove, "z"].iloc[0]
            ) / 2
            new_trace_data.loc[new_trace_data["Spot_ID"] == id_to_keep, "x"] = x
            new_trace_data.loc[new_trace_data["Spot_ID"] == id_to_keep, "y"] = y
            new_trace_data.loc[new_trace_data["Spot_ID"] == id_to_keep, "z"] = z
            new_trace_data.drop(
                new_trace_data[new_trace_data["Spot_ID"] == id_to_remove].index,
                inplace=True,
            )

        return new_trace_data, len(spot_id_to_keep)

    @staticmethod
    def remove_duplicates(trace_data):
        """For each individual trace, the duplicated barcodes are removed. If the remaining trace contains enough
        barcodes (above the minimal fraction p) the trace is saved, else it is discarded.

        @param trace_data: (pandas dataframe) input data with all the traces & detections
        @param p: (float) minimum fraction of barcodes required to keep a trace
        @return: (pandas dataframe) output a dataframe with the updated traces
        """
        new_trace_data = trace_data.copy()
        all_trace_id = trace_data["Trace_ID"].drop_duplicates()
        n_kept_w_correction = 0
        n_kept_wo_correction = 0
        n_discarded = 0

        for trace_id in tqdm(all_trace_id):
            single_trace = trace_data.loc[trace_data["Trace_ID"] == trace_id]
            trace_bc = single_trace["Barcode #"].tolist()
            unique_trace_bc = single_trace["Barcode #"].drop_duplicates().tolist()

            # check if there is duplicated barcodes in the trace. If yes, compute the list of repeated barcodes
            if len(unique_trace_bc) != len(trace_bc):
                repeated_bc = [bc for bc in unique_trace_bc if trace_bc.count(bc) > 1]
            else:
                repeated_bc = []

            # if the length of the trace after removing the duplicated barcodes is above the threshold, it is kept.
            # Else, the trace is discarded.
            if (len(unique_trace_bc) - len(repeated_bc)) > 1:
                if repeated_bc:
                    n_kept_w_correction += 1
                    for bc in repeated_bc:
                        if len(trace_data[trace_data["Barcode #"] == bc]) > 1:
                            new_trace_data.drop(
                                new_trace_data[
                                    (new_trace_data["Barcode #"] == bc)
                                    & (new_trace_data["Trace_ID"] == trace_id)
                                ].index,
                                inplace=True,
                            )
                else:
                    n_kept_wo_correction += 1
            else:
                n_discarded += 1
                new_trace_data.drop(
                    new_trace_data[new_trace_data["Trace_ID"] == trace_id].index,
                    inplace=True,
                )

        return new_trace_data, [n_discarded, n_kept_w_correction, n_kept_wo_correction]

    def save_to_astropy(self, trace_data, tag=None):
        """save panda dataframe into astropy table

        @param trace_data: (pd dataframe) contains all the traces
        @param tag: (str) indicate the tag to add to the filename
        """
        if tag is None:
            tag = "_bck"

        outputfile = (
            self.dest_folder + os.sep + self.data_file.split(".")[0] + tag + ".ecsv"
        )

        trace_data = Table.from_pandas(trace_data)
        trace_data.write(outputfile, overwrite=True)

        return outputfile

    def save_individual_labels(self, trace, tag=None):
        """helper function used to sort individual traces based on label value and save them in individual ecsv file.

        @param trace: (pd dataframe) input trace data
        @param tag: (str) tag to add to all individual files
        """
        unique_labels = trace["label"].drop_duplicates()
        for label in unique_labels:
            if tag:
                self.save_to_astropy(
                    trace[trace["label"] == label], tag=tag + "_" + label
                )
            else:
                self.save_to_astropy(trace[trace["label"] == label], tag=label)

    @staticmethod
    def him_map_2d_to_1d(map_2d):
        """Flatten a him 2d-map (either contact or distance) into a single vector. Since the map is symmetric along the
        first diagonal, only the first half is kept.

        @param map_2d: (numpy array) 2d him map
        @return: (numpy array) distance_flatten as a 1-dimension vector
        """
        distance_flatten = []
        n_bins = map_2d.shape[0]
        for x in range(n_bins - 1):
            distance_flatten.append(map_2d[x, x + 1 : n_bins].tolist())

        distance_flatten = [item for sublist in distance_flatten for item in sublist]
        return distance_flatten


if __name__ == "__main__":
    # [parsing arguments]
    p = parse_arguments()

    # parameters for the data - from the HiM astropy output files.
    # ------------------------------------------------------------
    data_folder = p["rootFolder"]
    data_files = p["trace_files"]
    dest_folder = data_folder + os.sep + p["output"]
    overlapping_threshold = p["overlapping_threshold"]  # in µm
    fraction_missing_barcodes = p[
        "fraction_missing_barcodes"
    ]  # a fraction of 0.5 means that a maximum of 50% missing barcodes is allowed

    print(f"\n$ Will process the following trace files: {data_files}\n")
    create_folder(dest_folder)

    for file in data_files:
        print(f"$ processing{file}")
        # instantiate the class
        # ---------------------
        _trace = FilterTraces(
            data_folder, file, dest_folder, threshold=overlapping_threshold
        )
        n_traces_total = _trace.data.shape[0]

        # perform a "hard" filtering where all the traces with at least one duplicated barcode are discarded
        # --------------------------------------------------------------------------------------------------
        hard_filtered_traces = _trace.hard_filtering()
        _trace.save_to_astropy(
            hard_filtered_traces, tag="_hard_filtered"
        )  # saving_path

        # select the traces with no duplicates (and at least 2 barcodes) and calculate the distance threshold used
        # later to filter the traces. Plot also the statistics for the selected traces
        # --------------------
        trace_wo_duplicates = _trace.select_traces_wo_duplicates(
            _trace.data, N_barcodes=p["N_barcodes"]
        )
        if len(trace_wo_duplicates) > 0:
            pwd_min, pwd_max = _trace.calculate_pwd_threshold(
                trace_wo_duplicates, verbose=True, save=True
            )
            print(
                f"\n$ Initial analysis returned {len(trace_wo_duplicates)} out of {n_traces_total} traces with no duplicates & at "
                f'least {p["N_barcodes"]} barcodes'
            )
            _trace.trace_statistics(save=True)

            # based on the threshold defines above, analyze each trace in order to remove unspecific detections & separate
            # traces that have been clustered together
            # ----------------------------------------
            filtered_traces = _trace.filter_traces(verbose=False)
            trace_wo_duplicates = _trace.select_traces_wo_duplicates(
                filtered_traces, N_barcodes=p["N_barcodes"]
            )
            print(
                f"\n$ After filtering, {len(trace_wo_duplicates)} traces are found with no duplicates & at least 2 barcodes"
            )

            # filter the traces by removing duplicated barcodes and checking the number of barcodes is above the threshold
            # mentioned in the parameters
            # ---------------------------
            filtered_traces, n_overlapping_bc = _trace.detect_overlapping_barcodes(
                filtered_traces, verbose=True, save=True
            )
            trace_wo_duplicates = _trace.select_traces_wo_duplicates(
                filtered_traces, N_barcodes=p["N_barcodes"]
            )
            print(
                f"\n$ A total of {n_overlapping_bc} overlapping barcodes were found. In total, {len(trace_wo_duplicates)} "
                f"traces are now found without duplicates and at least 2 barcodes"
            )

            final_traces, stat = _trace.remove_duplicates(filtered_traces)
            print(
                f"\n$ After completion of the procedure: \n\t {stat[0]} traces were discarded \n\t {stat[1]} traces with "
                f"duplicated barcodes were kept \n\t {stat[2]} were found without any duplicates."
            )

            # saving_name = dest_folder + os.sep + file.split('.')[0] + '_filtered.ecsv'
            # saving_path = os.path.join(dest_folder, saving_name)
            outputfile = _trace.save_to_astropy(final_traces, tag="_" + p["output"])
            print(f"\n$ output trace file: {outputfile}")
        else:
            print("! Sorry, all traces seem to contain duplicate barcodes")
