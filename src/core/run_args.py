#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyHiM argument parser module"""


import os
from argparse import ArgumentParser

from core.pyhim_logging import print_log


def _parse_run_args(command_line_arguments):
    """Parse run arguments of pyHiM runtime

    Parameters
    ----------
    command_line_arguments : List[str]
        Used to give inputs for the runtime when you call this function like a module.
        For example, to test the pyHiM run from tests folder.

    Returns
    -------
    ArgumentParser.args
        An accessor of run arguments
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-C",
        "--cmd",
        help="Comma-separated list of routines to run.\nDEFAULT: project,register_global,register_local,mask_2d,localize_2d,mask_3d,localize_3d,filter_localizations,register_localizations,build_traces,build_matrix",
    )
    # Data path
    parser.add_argument(
        "-F",
        "--rootFolder",
        type=str,
        default=os.getcwd(),
        help="Folder path with input images.\nDEFAULT: Current directory",
    )

    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        default=None,
        help="Path of the parameters.json file.\nDEFAULT: pyHiM expect this file inside your current directory",
    )
    # Number of threads
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=1,
        help="Thread number to run with parallel mode.\nDEFAULT: 1 (sequential mode)",
    )

    return parser.parse_args(command_line_arguments)


class RunArgs:
    """Store and check run arguments"""

    def __init__(self, command_line_arguments):
        print_log("\n-----------------------------------------------------------------")
        parsed_args = _parse_run_args(command_line_arguments)
        self.data_path = (
            parsed_args.rootFolder if parsed_args.rootFolder != "." else os.getcwd()
        )
        self._is_docker()
        self.cmd_list = self.parse_cmd(parsed_args.cmd)
        self.thread_nbr = parsed_args.threads
        self.parallel = self.thread_nbr > 1
        self.params_path = parsed_args.parameters
        self._check_consistency()

    def _is_docker(self):
        """Change the data path is run inside docker"""
        if "docker" in os.environ:
            self.data_path = "/data"
            print_log(f"\n\n$ Running in docker, him_data: {self.data_path}")

    def _check_consistency(self):
        if not os.path.isdir(self.data_path):
            raise SystemExit(f"Input data path ({self.data_path}) NOT FOUND.")

        available_commands = self.get_available_commands()
        low_available_cmds = [a_c.lower() for a_c in available_commands]
        for cmd in self.cmd_list:
            if cmd.lower() not in low_available_cmds:
                print_log(
                    f"\n\n# ERROR: {cmd} isn't an available command like:\n\
                    {available_commands}\n",
                    status="WARN",
                )
                raise SystemExit(
                    f"'{cmd}' isn't an available command like:\n\
                    {available_commands}"
                )

        if not isinstance(self.thread_nbr, int) or self.thread_nbr < 1:
            raise SystemExit(f"Number of threads ({self.thread_nbr}): INVALID.")

        if self.params_path is not None and not os.path.isfile(self.params_path):
            raise SystemExit(f"Input parameters file ({self.data_path}) NOT FOUND.")

    def args_to_str(self):
        """Print parameters in your shell terminal

        Parameters
        ----------
        dictionary : dict
            Parameters dictionary
        """

        def tab_spacer(name: str, val: str):
            spacer = "\t" * (3 - int(len(name) / 8))
            return "\t" + name + spacer + val + "\n"

        to_print = "\n$ Loaded arguments:\n"
        to_print += tab_spacer("rootFolder", str(self.data_path))
        if self.params_path:
            to_print += tab_spacer("parameters", str(self.params_path))
        to_print += tab_spacer("threads", str(self.thread_nbr))
        to_print += tab_spacer("parallel", str(self.parallel))
        to_print += tab_spacer("cmd", str(self.cmd_list))

        return to_print

    @classmethod
    def parse_cmd(cls, cmd):
        """Parse the input command list give by the user as a string (comma-separated)

        Parameters
        ----------
        cmd : str
            A comma-separated human-list of commands

        Returns
        -------
        List[str]
            A Python list of commands
        """
        if cmd:
            if cmd == "2D":
                return cls.get_2d_commands()
            elif cmd == "3D":
                return cls.get_3d_commands()
            else:
                return cmd.split(",")
        # else by default give every commands:
        return cls.get_available_commands()

    @staticmethod
    def get_available_commands():
        """Available commands for pyHiM

        Returns
        -------
        frozenset
            Set of available commands
        """
        return (
            "project",
            "register_global",
            "register_local",
            "mask_2d",
            "localize_2d",
            "mask_3d",
            "localize_3d",
            "makeProjections",  # DEPRECATED
            "appliesRegistrations",  # DEPRECATED
            "alignImages",  # DEPRECATED
            "alignImages3D",  # DEPRECATED
            "segmentMasks",  # DEPRECATED
            "segmentMasks3D",  # DEPRECATED
            "segmentSources3D",  # DEPRECATED
            "shift_mask",
            "filter_localizations",
            "register_localizations",
            "build_traces",
            "build_matrix",
        )

    @staticmethod
    def get_2d_commands():
        """Default commands for 2D pipeline

        Returns
        -------
        frozenset
            Set of 2D commands
        """
        return (
            "project",
            "register_global",
            "mask_2d",
            "localize_2d",
            "filter_localizations",
            "build_traces",
            "build_matrix",
        )

    @staticmethod
    def get_3d_commands():
        """Default commands for 3D pipeline

        Returns
        -------
        frozenset
            Set of 3D commands
        """
        return (
            "project",
            "register_global",
            "register_local",
            "mask_3d",
            "localize_3d",
            "filter_localizations",
            "register_localizations",
            "build_traces",
            "build_matrix",
        )
