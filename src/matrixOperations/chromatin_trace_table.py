#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:33:57 2022

@author: marcnol

trace table management class

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apifish.stack.io import read_table_from_ecsv, save_table_to_ecsv
from astropy.table import Table, vstack
from stardist import random_label_cmap
from tqdm import tqdm

from imageProcessing.localization_table import (
    build_color_dict,
    decode_rois,
    plots_localization_projection,
)

lbl_cmap = random_label_cmap()
font = {"weight": "normal", "size": 22}
matplotlib.rc("font", **font)


class ChromatinTraceTable:

    def __init__(self, xyz_unit="micron", genome_assembly="mm10"):
        """
        Initializes the ChromatinTraceTable class.
        """
        self.a = 1
        self.xyz_unit = xyz_unit
        self.genome_assembly = genome_assembly
        self.experimenter_name = ""
        self.experimenter_contact = ""
        self.software_title = ""
        self.software_authors = ""
        self.lab_name = ""
        self.software_description = ""
        self.software_repository = ""
        self.software_citation = ""
        self.columns = []
        self.data = None
        self.original_format = "ecsv"  # Default format

    def initialize(self):
        self.data = Table(
            names=(
                "Spot_ID",
                "Trace_ID",
                "x",
                "y",
                "z",
                "Chrom",
                "Chrom_Start",
                "Chrom_End",
                "ROI #",
                "Mask_id",
                "Barcode #",
                "label",
            ),
            dtype=(
                "S2",
                "S2",
                "f4",
                "f4",
                "f4",
                "S2",
                "int",
                "int",
                "int",
                "int",
                "int",
                "S2",
            ),
        )

        self.data.meta["comments"] = [
            f"xyz_unit={self.xyz_unit}",
            f"genome_assembly={self.genome_assembly}",
        ]

    def load(self, file):
        """
        Loads a trace table from a .ecsv or .4dn file.
        """
        if not os.path.exists(file):
            print(f"# ERROR: could not find file: {file}")
            sys.exit()

        file_ext = os.path.splitext(file)[1].lower()
        if file_ext == ".ecsv":
            print("$ Importing table from pyHiM format")
            self.data = read_table_from_ecsv(file)
            self.original_format = "ecsv"
        elif file_ext == ".4dn":
            print("$ Importing table from fof-ct format")
            self._read_metadata_from_4dn(file)
            self.data = self._convert_4dn_to_astropy(file)
            self.original_format = "4dn"
        else:
            raise ValueError("Unsupported file format. Use .ecsv or .4dn")

        print(f"Successfully loaded trace table: {file}")
        return self.data

    def save(self, file_name, table, comments=""):
        """
        Saves the trace table in the appropriate format (.ecsv or .4dn).
        """
        if self.original_format == "4dn":
            self._convert_astropy_to_4dn(self.data, file_name)
        else:
            print(f"$ Saving output table as {file_name} ...")

            try:
                table.meta["comments"].append(comments)
            except KeyError:
                table.meta["comments"] = [comments]

            save_table_to_ecsv(table, file_name)

    def _read_metadata_from_4dn(self, file):
        """
        Reads metadata fields from a .4dn file and stores them as class attributes.
        """
        with open(file, "r") as f:
            for line in f:
                if line.startswith("#experimenter_name:"):
                    self.experimenter_name = line.split(": ")[1].strip()
                elif line.startswith("#experimenter_contact:"):
                    self.experimenter_contact = line.split(": ")[1].strip()
                elif line.startswith("##genome_assembly:"):
                    self.genome_assembly = line.split("=")[1].strip()
                elif line.startswith("#Software_Title:"):
                    self.software_title = line.split(": ")[1].strip()
                elif line.startswith("#Software_Authors:"):
                    self.software_authors = line.split(": ")[1].strip()
                elif line.startswith("#lab_name:"):
                    self.lab_name = line.split(": ")[1].strip()
                elif line.startswith("#Software_Description:"):
                    self.software_description = line.split(": ")[1].strip()
                elif line.startswith("#Software_Repository:"):
                    self.software_repository = line.split(": ")[1].strip()
                elif line.startswith("#Software_PreferredCitationID:"):
                    self.software_citation = line.split(": ")[1].strip()
                elif line.startswith("##columns="):
                    self.columns = line.split("=")[1].split("(")[1].split(")")[0]
                    self.columns = self.columns.split(", ")
                    # self.columns = self._read_column_names_from_4dn(file)
                    print(f"> Columns read: {self.columns}")

    def _convert_4dn_to_astropy(self, fofct_file):
        """
        Converts a .4dn file to an Astropy table with appropriate formatting.
        Also saves a BED file mapping genomic coordinates to barcode numbers.
        """
        column_names = self.columns  # self._read_column_names_from_4dn(fofct_file)
        csv_data = pd.read_csv(fofct_file, comment="#", header=None, names=column_names)

        # Rename XYZ columns for Astropy compatibility
        csv_data.rename(columns={"X": "x", "Y": "y", "Z": "z"}, inplace=True)

        # Handle optional Cell_ID column
        if "Cell_ID" in csv_data.columns:
            csv_data.rename(columns={"Cell_ID": "Mask_id"}, inplace=True)
        else:
            print("> No Cell_ID column found, will use -1 for Mask_id")
            csv_data["Mask_id"] = -1  # Placeholder if Cell_ID is missing

        # Handle optional Extra_Cell_ROI_ID column
        if "Extra_Cell_ROI_ID" in csv_data.columns:
            csv_data.rename(columns={"Extra_Cell_ROI_ID": "ROI #"}, inplace=True)
        else:
            print("> No Extra_Cell_ROI_ID column found, will use 0 for Mask_id")
            csv_data["ROI #"] = 0  # Default value if missing

        # Assign Barcode # by ordering and mapping unique genomic positions
        unique_barcodes = (
            csv_data[["Chrom", "Chrom_Start", "Chrom_End"]]
            .drop_duplicates()
            .sort_values(by=["Chrom", "Chrom_Start", "Chrom_End"])
            .reset_index(drop=True)
        )
        unique_barcodes["Barcode #"] = range(1, len(unique_barcodes) + 1)
        barcode_mapping = {
            tuple(row[:3]): row[3]
            for row in unique_barcodes.itertuples(index=False, name=None)
        }

        csv_data["Barcode #"] = csv_data.apply(
            lambda row: barcode_mapping[
                (row["Chrom"], row["Chrom_Start"], row["Chrom_End"])
            ],
            axis=1,
        )
        csv_data["label"] = "None"  # Placeholder for label

        # Save BED file with Barcode # mapping
        bed_file = fofct_file.replace(".4dn", ".bed")
        unique_barcodes.to_csv(bed_file, sep="\t", header=False, index=False)
        print(f"Saved BED file: {bed_file}")

        return Table.from_pandas(csv_data)

    def _convert_astropy_to_4dn(self, table, output_file):
        """
        Converts an Astropy table back to .4dn format with appropriate headers.
        """
        output_file = output_file.strip(".ecsv") + ".4dn"

        csv_data = table.to_pandas()
        csv_data.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

        # Remove extra columns
        csv_data = csv_data.drop(columns=["Barcode #", "label"], errors="ignore")

        # Rename columns
        if "Extra_Cell_ROI_ID" in self.columns:
            csv_data.rename(columns={"ROI #": "Extra_Cell_ROI_ID"}, inplace=True)
        else:
            csv_data = csv_data.drop(columns=["ROI #"], errors="ignore")

        if "Cell_ID" in self.columns:
            csv_data.rename(columns={"Mask_id": "Cell_ID"}, inplace=True)
        else:
            csv_data = csv_data.drop(columns=["Mask_id"], errors="ignore")

        # parses column list for header
        column_list = ", ".join(self.columns)

        header = f"""##FOF-CT_version=v0.1
##Table_namespace=4dn_FOF-CT_core
##genome_assembly={self.genome_assembly}
##XYZ_unit=micron
#Software_Title: {self.software_title}
#Software_Type: SpotLoc+Tracing
#Software_Authors: {self.software_authors}
#Software_Description: {self.software_description}
#Software_Repository: {self.software_repository}
#Software_PreferredCitationID: {self.software_citation}
#lab_name: {self.lab_name}
#experimenter_name: {self.experimenter_name}
#experimenter_contact: {self.experimenter_contact}
#additional_tables:
##columns=({column_list})
"""
        # print(f"> Columnds to export to 4dn table: {csv_data.columns}")
        with open(output_file, "w") as f:
            f.write(header)
            csv_data.to_csv(f, index=False, header=False)
        print(f"Saved 4dn trace table with headers: {output_file}")

    def load_bed_file(self, bed_file):
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
                    print(
                        f"Warning: Skipping line with invalid data types: {line.strip()}"
                    )
                    print(f"Error: {e}")

        if not bed_dict:
            raise ValueError("No valid entries found in the BED file.")

        print(f"Successfully loaded {len(bed_dict)} barcode mappings from BED file.")

        return bed_dict

    def impute_genomic_coordinates(self, bed_dict, auto_continue=False):
        """Updates the Chrom, Chrom_Start, and Chrom_End columns in the trace file based on the BED file."""

        unmatched_barcodes = set()
        matched_count = 0
        total_count = 0

        for row in self.data:
            barcode = row["Barcode #"]
            total_count += 1

            if barcode in bed_dict:
                row["Chrom"] = bed_dict[barcode]["Chrom"]
                row["Chrom_Start"] = bed_dict[barcode]["Chrom_Start"]
                row["Chrom_End"] = bed_dict[barcode]["Chrom_End"]
                matched_count += 1
            else:
                unmatched_barcodes.add(barcode)

        if unmatched_barcodes:
            missing_percent = (len(unmatched_barcodes) / total_count) * 100
            print(
                f"Warning: {len(unmatched_barcodes)} unique barcodes ({missing_percent:.1f}% of rows) couldn't be matched:"
            )

            # Only show up to 10 unmatched barcodes to avoid cluttering the output
            if len(unmatched_barcodes) <= 10:
                print(
                    f"  Unmatched barcodes: {', '.join(map(str, sorted(unmatched_barcodes)))}"
                )
            else:
                print(
                    f"  First 10 unmatched barcodes: {', '.join(map(str, sorted(list(unmatched_barcodes))[:10]))}"
                )
                print(f"  ... and {len(unmatched_barcodes) - 10} more")

            # Ask the user if they want to continue if more than 10% of barcodes are unmatched
            if missing_percent > 10 and ~auto_continue:
                response = input(
                    "More than 10% of barcodes couldn't be matched. Continue anyway? (y/n): "
                )
                if response.lower() != "y":
                    print("Operation aborted by user.")
                    return

        print(
            f"Successfully matched {matched_count}/{total_count} rows ({(matched_count/total_count)*100:.1f}%)"
        )

        return matched_count, total_count

    def append(self, table):
        """
        appends <table> to self.data

        Parameters
        ----------
        table : astropy table
            table to append to existing self.data table.

        Returns
        -------
        None.

        """

        self.data = vstack([self.data, table])

    def filter_traces_by_coordinate(self, coor="z", coor_min=0.0, coor_max=np.inf):
        """
        This function will remove the spots that are outside coordinate limits

        Parameters
        ----------
        coor : string, optional
            which coordinate to process ('x','y' or 'z'). The default is 'z'.
        coor_min : float, optional
            minimum value. The default is 0..
        coor_max : float, optional
            maximum value. The default is np.inf.

        Returns
        -------
        updated trace table is kept in self.data

        """
        trace_table = self.data

        if len(trace_table) > 0:
            # indexes trace file
            trace_table_indexed = trace_table.group_by("Trace_ID")

            # iterates over traces
            print(f"\n$ Will keep localizations with {coor_min} < {coor} < {coor_max}.")
            print(
                f"$ Number of original spots / traces: {len(trace_table)} / {len(trace_table_indexed.groups)}"
            )
            rows_to_remove = []
            for idx, row in enumerate(trace_table):
                coordinate = float(row[coor])

                if coordinate < coor_min or coordinate > coor_max:
                    rows_to_remove.append(idx)
                    # coordinates.append(coordinate)

            print(f"$ Number of spots to remove: {len(rows_to_remove)}")

            trace_table.remove_rows(rows_to_remove)

            if len(trace_table) > 0:
                trace_table_indexed = trace_table.group_by("Trace_ID")
                number_traces_left = len(trace_table_indexed.groups)
            else:
                number_traces_left = 0

            print(
                f"$ Number of spots / traces left: {len(trace_table)} / {number_traces_left}"
            )

        else:
            print("! Error: you are trying to filter an empty trace table!")
        self.data = trace_table

    def filter_by_intensity(self, trace, localizations, intensity_min):
        """
        Filters localizations in the trace file based on intensity from the localization table.
        """
        localizations.add_index("Buid")  # Add an index for fast lookup

        rows_to_remove = []
        number_spots = len(trace.data)
        intensities_kept = list()
        for idx, row in enumerate(trace.data):
            spot_id = row["Spot_ID"]
            try:
                intensity = localizations.loc[spot_id]["peak"]
                if intensity < intensity_min:
                    rows_to_remove.append(idx)
                else:
                    intensities_kept.append(intensity)
            except KeyError:
                continue  # If Spot_ID is not found, keep the entry

        trace.data.remove_rows(rows_to_remove)
        print(
            f"> Removed {len(rows_to_remove)}/{number_spots} localizations below intensity threshold ({intensity_min})."
        )
        print(f"> Number of rows in filtered trace table: {len(trace.data)}")

        return intensities_kept

    def barcode_statistics(self, trace_table):
        """
        calculates the number of times a barcode is repeated in a trace for all traces in a trace table

        Parameters
        ----------
        trace_table : ASTROPY table
            trace table.

        Returns
        -------
        collective_barcode_stats : dict
            dict with barcode identities as keys and a list of the number of times it was present in each trace treated.

        """
        collective_barcode_stats = {}

        trace_table_indexed = trace_table.group_by("Trace_ID")

        # iterates over traces
        print("$ Calculating barcode stats...")

        for trace in tqdm(trace_table_indexed.groups):
            unique_barcodes = list(set(trace["Barcode #"].data))
            barcodes = list(trace["Barcode #"].data)

            # if number_unique_barcodes < number_barcodes:
            barcode_stats = {}
            for barcode in unique_barcodes:
                barcode_rep = barcodes.count(barcode)
                barcode_stats[str(barcode)] = barcode_rep

                if str(barcode) in collective_barcode_stats:
                    collective_barcode_stats[str(barcode)].append(barcode_rep)
                else:
                    collective_barcode_stats[str(barcode)] = [barcode_rep]

        return collective_barcode_stats

    def plots_barcode_statistics(
        self,
        collective_barcode_stats,
        file_name="barcode_stats",
        kind="violin",
        norm=True,
    ):
        """
        plots the collecive_bracode stats (see previous function)

        Parameters
        ----------
        collective_barcode_stats : dict
            dict with barcode identities as keys and a list of the number of times it was present in each trace treated.
        file_name : str, optional
            output filename for saving figure. The default is 'barcode_stats.png'.
        kind : str, optional
            Options for plotting styles: 'violin' or 'matrix'. The default is 'violin'.

        Returns
        -------
        None.

        """
        sorted_barcodes = sorted([int(x) for x in collective_barcode_stats.keys()])
        data = [collective_barcode_stats[str(key)] for key in sorted_barcodes]

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

        label, density = ("frequency", True) if norm else ("counts", False)
        ax1.set_title("Distribution of barcodes per trace")

        if "violin" in kind:
            self._extracted_from_plots_barcode_statistics_38(ax1, data, sorted_barcodes)
        else:
            bins = range(1, 10)
            matrix = np.zeros((len(sorted_barcodes), len(bins) - 1))
            for idx, barcode_data in enumerate(data):
                matrix[idx, :], _ = np.histogram(
                    barcode_data, bins=bins, density=density
                )
            bin_number = list(bins)
            pos = ax1.imshow(np.transpose(matrix), cmap="Reds")
            ax1.set_xticks(np.arange(matrix.shape[0]), sorted_barcodes)
            ax1.set_yticks(np.arange(0, len(bins)), bin_number)
            ax1.set_ylabel("number of barcodes")
            ax1.set_xlabel("barcode id")
            fig.colorbar(
                pos, ax=ax1, location="bottom", anchor=(0.5, 1), shrink=0.4, label=label
            )

        fig.savefig(f"{file_name}.png")

    # TODO Rename this here and in `plots_barcode_statistics`
    def _extracted_from_plots_barcode_statistics_38(self, ax1, data, sorted_barcodes):
        ax1.set_ylabel("number of barcodes")
        ax1.violinplot(data)

        ax1.set_xticks(np.arange(1, len(sorted_barcodes) + 1), labels=sorted_barcodes)
        ax1.set_xlim(0.25, len(sorted_barcodes) + 0.75)
        ax1.set_ylim(0.0, 10)
        ax1.set_xlabel("barcode id")

    def trace_remove_label(self, label=""):
        """
        This function will remove traces that do not contain the word 'label' in the 'label' column

        Parameters
        ----------
        label : TYPE, string
            the labe to keep. The default is "".

        Returns
        -------
        None.

        """
        trace_table = self.data

        trace_table_new = trace_table.copy()

        rows_to_remove = []

        for idx, row in enumerate(trace_table_new):
            if label in row["label"]:
                rows_to_remove.append(idx)

        trace_table_new.remove_rows(rows_to_remove)

        removed = len(trace_table) - len(trace_table_new)
        print(f"$ Removed {removed} spots that contained the label: {label}")

        self.data = trace_table_new

    def trace_keep_label(self, label=""):
        """
        This function will remove traces that do not contain the word 'label' in the 'label' column

        Parameters
        ----------
        label : TYPE, string
            the labe to keep. The default is "".

        Returns
        -------
        None.

        """
        trace_table = self.data

        trace_table_new = trace_table.copy()

        rows_to_remove = []

        for idx, row in enumerate(trace_table_new):
            if label not in row["label"]:
                rows_to_remove.append(idx)

        trace_table_new.remove_rows(rows_to_remove)

        removed = len(trace_table) - len(trace_table_new)
        print(f"$ Removed {removed} spots that did not contain the label: {label}")

        self.data = trace_table_new

    def filter_repeated_barcodes(self, trace_file="mock"):
        """
        This function will remove the barcodes that are present more than once in a trace.
        All other barcodes are kept.

        Parameters
        ----------


        Returns
        -------
        updated trace table is kept in self.data

        """
        trace_table = self.data
        trace_table_new = trace_table.copy()
        print("\n$ Removing spots with repeated barcodes...")
        if len(trace_table) > 0:
            # indexes trace file
            trace_table_indexed = trace_table.group_by("Trace_ID")

            # iterates over traces
            print(
                f"\n$ Number of original \n spots: {len(trace_table)} \n traces: {len(trace_table_indexed.groups)}"
            )

            # calculates the statistics for the table before processing
            collective_barcode_stats = self.barcode_statistics(trace_table)

            # plots statistics of barcodes and saves in file
            self.plots_barcode_statistics(
                collective_barcode_stats,
                file_name=f"{trace_file}_before",
                kind="matrix",
                norm=True,
            )

            # iterates over traces
            spots_to_remove = []
            for trace in tqdm(trace_table_indexed.groups):
                unique_barcodes = list(set(trace["Barcode #"].data))
                number_unique_barcodes = len(unique_barcodes)
                barcodes = list(trace["Barcode #"].data)
                number_barcodes = len(barcodes)

                if number_unique_barcodes < number_barcodes:
                    trace_indexed_by_barcode = trace.group_by("Barcode #")

                    for row in trace_indexed_by_barcode:
                        barcode = row["Barcode #"].data
                        barcode_rep = barcodes.count(barcode)

                        # if a barcode is more than once I will remove both instances
                        if barcode_rep > 1:
                            spots_to_remove.append(row["Spot_ID"])

            print(f"$ Number of spots to remove: {len(spots_to_remove)}")
            print("$ Removing repeated spots...")

            rows_to_remove = []
            for idx, row in enumerate(trace_table):
                spot_id = row["Spot_ID"]

                if spot_id in spots_to_remove:
                    rows_to_remove.append(idx)

            trace_table_new.remove_rows(rows_to_remove)

            print(f"$ Number of rows to remove: {len(rows_to_remove)}")

            if len(trace_table_new) > 0:
                trace_table_indexed = trace_table_new.group_by("Trace_ID")
            #     number_traces_left = len(trace_table_indexed.groups)
            # else:
            #     number_traces_left = 0

            print(
                f"$ After filtering, I see \n spots: {len(trace_table_new)} \n traces: {len(trace_table_indexed.groups)}"
            )

            # calculates the statistics for the table before processing
            collective_barcode_stats_new = self.barcode_statistics(trace_table_new)

            # plots statistics of barcodes and saves in file
            self.plots_barcode_statistics(
                collective_barcode_stats_new,
                file_name=f"{trace_file}_filtered",
                kind="matrix",
                norm=False,
            )

        else:
            print("! Error: you are trying to filter an empty trace table!")
        self.data = trace_table_new

    def remove_duplicates(
        self,
    ):  # sourcery skip: extract-method
        """
        removes duplicated (identical) spots

        Parameters
        ----------


        Returns
        -------
        trace_table : ASTROPY Table
            output trace table.
        """
        trace_table = self.data
        trace_table_new = trace_table.copy()
        print("\n$ Removing duplicated barcodes...")
        if len(trace_table) > 0:
            # indexes trace file
            trace_table_indexed = trace_table.group_by("Spot_ID")

            # finds barcodes with the same UID and stores UIDs in list
            spots_to_remove = [
                trace["Spot_ID"][0]
                for trace in tqdm(trace_table_indexed.groups)
                if len(trace) > 1
            ]

            # finds row of the first offending barcode
            # this only removes one of the duplicated barcodes --> assumes at most there are two copies
            rows_to_remove = []
            for idx, row in enumerate(trace_table):
                spot_id = row["Spot_ID"]
                if spot_id in spots_to_remove:
                    rows_to_remove.append(idx)
                    spots_to_remove.remove(spot_id)

            # removes from table
            trace_table_new.remove_rows(rows_to_remove)

            print(f"$ Number of rows to remove: {len(rows_to_remove)}")

            if len(trace_table_new) > 0:
                trace_table_indexed = trace_table_new.group_by("Trace_ID")
                number_traces_left = len(trace_table_indexed.groups)
            else:
                number_traces_left = 0

            print(
                f"$ After filtering, I see \n spots: {len(trace_table_new)} \n traces: {number_traces_left}"
            )

        else:
            print("! Error: you are trying to filter an empty trace table!")

        self.data = trace_table_new

    def remove_barcode(self, remove_barcode=None):
        """
        Removes a specific barcode from a trace table

        Returns
        -------
        trace_table : ASTROPY Table
            output trace table.
        """

        if remove_barcode is not None:
            print(f"$ Removing barcode <{remove_barcode}>")

            trace_table = self.data
            trace_table_new = trace_table.copy()

            # indexes trace file
            trace_table_indexed = trace_table.group_by("Barcode #")
            number_barcodes_before = len(trace_table_indexed.groups)

            # iterates over traces
            spots_to_remove = []
            for sub_table_barcode in trace_table_indexed.groups:
                barcode_name = list(set(sub_table_barcode["Barcode #"]))
                if int(remove_barcode) in barcode_name:
                    print(f"$ Found barcode: {barcode_name}")
                    spots_to_remove.extend(row["Spot_ID"] for row in sub_table_barcode)
            print(f"$ Number of spots to remove: {len(spots_to_remove)}")

            # builds the list with the rows to remove
            rows_to_remove = []
            for idx, row in enumerate(tqdm(trace_table)):
                spot_id = row["Spot_ID"]

                if spot_id in spots_to_remove:
                    rows_to_remove.append(idx)

            # removes targeted spots
            trace_table_new.remove_rows(rows_to_remove)

            # provides statistics
            trace_table_indexed_new = trace_table_new.group_by("Barcode #")
            number_barcodes_left = len(trace_table_indexed_new.groups)
            print(
                f"\n$ Number of barcodes \n\t original: {number_barcodes_before} \n\t after: {number_barcodes_left}"
            )

        self.data = trace_table_new

    def filter_traces_by_n(self, minimum_number_barcodes=2):
        """
        Removes rows in trace table with less than `minimum_number_barcodes` barcodes

        Parameters
        ----------
        trace_table : ASTROPY Table
            input trace table.
        minimum_number_barcodes : TYPE, optional
            minimum number of barcodes in trace. The default is 1.

        Returns
        -------
        trace_table : ASTROPY Table
            output trace table.

        """

        trace_table = self.data

        # indexes trace file
        trace_table_indexed = trace_table.group_by("Trace_ID")

        # iterates over traces
        print(f"\n$ Removing traces with < {minimum_number_barcodes} spots")
        print(
            f"$ Number of original spots / traces: {len(trace_table)} / {len(trace_table_indexed.groups)}"
        )

        barcodes_to_remove = []
        print("$ Analyzing traces...")

        for trace in tqdm(trace_table_indexed.groups):
            number_unique_barcodes = len(list(set(trace["Barcode #"].data)))

            if number_unique_barcodes < minimum_number_barcodes:
                barcodes_to_remove.append(list(trace["Spot_ID"].data))

        print(f"$ Number of traces to remove: {len(barcodes_to_remove)}")

        list_barcode_to_remove = []
        for barcodes in tqdm(barcodes_to_remove):
            list_barcode_to_remove.extend(iter(barcodes))
        rows_to_remove = []

        print("$ Finding which rows to remove...")
        for idx, row in enumerate(tqdm(trace_table)):
            spot_id = row["Spot_ID"]
            if spot_id in list_barcode_to_remove:
                rows_to_remove.append(idx)

        trace_table.remove_rows(rows_to_remove)
        if len(trace_table) > 0:
            trace_table_indexed = trace_table.group_by("Trace_ID")
            number_traces_left = len(trace_table_indexed.groups)
        else:
            number_traces_left = 0

        print(
            f"$ Number of spots / traces left: {len(trace_table)} / {number_traces_left}"
        )

        self.data = trace_table

    def plots_traces(
        self, filename_list, masks=np.zeros((2048, 2048)), pixel_size=None
    ):
        """
        This function plots 3 subplots (xy, xz, yz) with the localizations.
        One figure is produced per ROI.

        Parameters
        ----------

        filename_list: list
            filename
        """

        if pixel_size is None:
            pixel_size = [0.1, 0.1, 0.25]
        data = self.data

        # indexes table by ROI
        data_indexed, number_rois = decode_rois(data)

        im_size = 60
        print(f"> Will make plots for {number_rois} ROI(s)")
        for i_roi in range(number_rois):
            # creates sub Table for this ROI
            data_roi = data_indexed.groups[i_roi]
            n_roi = data_roi["ROI #"][0]
            print(f"> Plotting barcode localization map for ROI: {n_roi}")
            color_dict = build_color_dict(data_roi, key="Barcode #")

            # initializes figure
            fig = plt.figure(constrained_layout=False)
            fig.set_size_inches((im_size * 2, im_size))
            gs = fig.add_gridspec(2, 2)
            ax = [
                fig.add_subplot(gs[:, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 1]),
            ]

            # defines variables
            x = data_roi["x"]
            y = data_roi["y"]
            z = data_roi["z"]

            colors = [color_dict[str(x)] for x in data_roi["Barcode #"]]
            titles = ["Z-projection", "X-projection", "Y-projection"]

            # plots masks if available
            if len(masks.shape) == 3:
                masks = np.max(masks, axis=0)
            ax[0].imshow(masks, cmap=lbl_cmap, alpha=0.3)

            # calculates mean trace positions and sizes by looping over traces
            data_traces = data_roi.group_by("Trace_ID")
            color_dict_traces = build_color_dict(data_traces, key="Trace_ID")
            colors_traces = [color_dict_traces[str(x)] for x in data_traces["Trace_ID"]]
            cmap_traces = plt.cm.get_cmap("hsv", np.max(colors_traces))
            number_traces = len(colors_traces)

            print(f"$ Plotting {number_traces} traces...")
            for trace, color, trace_id in zip(
                data_traces.groups, colors_traces, data_traces.groups.keys
            ):
                # Sort by barcode number
                sorted_trace = trace[np.argsort(trace["Barcode #"])]

                # Extract coordinates
                x_trace = sorted_trace["x"].data / pixel_size[0]
                y_trace = sorted_trace["y"].data / pixel_size[1]
                z_trace = (
                    sorted_trace["z"].data / pixel_size[2]
                )  # If needed for other plots

                # Plot scatter points
                ax[0].scatter(
                    x_trace,
                    y_trace,
                    color=cmap_traces(color),
                    s=5,
                    label=f"Trace {trace_id}",
                    alpha=0.1,
                )

                # Plot line connecting the points in this trace
                ax[0].plot(x_trace, y_trace, color="k", linewidth=1, alpha=0.4)

                # Repeat for the other projections
                ax[1].scatter(x_trace, z_trace, color=cmap_traces(color), s=5)
                ax[1].plot(x_trace, z_trace, color="k", linewidth=1, alpha=0.4)

                ax[2].scatter(y_trace, z_trace, color=cmap_traces(color), s=5)
                ax[2].plot(y_trace, z_trace, color="k", linewidth=1, alpha=0.4)

            print(f"$ Pixel_size = {pixel_size}")
            # makes plot
            plots_localization_projection(
                x / pixel_size[0], y / pixel_size[1], ax[0], colors, titles[0]
            )
            plots_localization_projection(
                x / pixel_size[0], z / pixel_size[2], ax[1], colors, titles[1]
            )
            plots_localization_projection(
                y / pixel_size[1], z / pixel_size[2], ax[2], colors, titles[2]
            )

            fig.tight_layout()

            """
            for trace, color, trace_id in zip(
                data_traces.groups, colors_traces, data_traces.groups.keys
            ):

                # Plots polygons for each trace
                poly_coord = np.array(
                    [
                        (trace["x"].data) / pixel_size[0],
                        (trace["y"].data) / pixel_size[1],
                    ]
                ).T
                polygon = Polygon(
                    poly_coord,
                    closed=False,
                    fill=False,
                    edgecolor=cmap_traces(color),
                    linewidth=1,
                    alpha=1,
                )
                ax[0].add_patch(polygon) # this does not work so I commented it out
            """

            # saves output figure
            filename_list_i = filename_list.copy()
            filename_list_i.insert(-1, f"_ROI{str(n_roi)}")
            traces = "".join(filename_list_i)
            try:
                fig.savefig(traces)
            except ValueError:
                print(f"\nValue error while saving output figure with traces:{traces}")
