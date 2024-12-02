#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for pyHiM logging
"""

import logging
import os
from datetime import datetime

from dask.distributed.worker import logger

# Default level of dask logger is WARN
logger.setLevel(logging.INFO)


class Logger:
    def __init__(
        self, root_folder, parallel=False, session_name="HiM_analysis", init_msg=""
    ):
        self.m_root_folder = root_folder
        self.log_file = ""
        self.md_filename = ""
        # Keeps the messages to be print in log until this is possible
        self.init_msg = init_msg
        self.setup_md_file(session_name)
        self.setup_logger()
        print_log(self.init_msg)
        print_log(f"$ Started logging to: {self.log_file}")

    def setup_md_file(self, session_name: str = "HiM_analysis"):
        self.init_msg += f"\n$ root_folder: {self.m_root_folder}\n"
        begin_time = datetime.now()
        now = datetime.now()
        date_time = now.strftime("%d%m%Y_%H%M%S")

        self.log_file = self.m_root_folder + os.sep + session_name + date_time + ".log"
        self.md_filename = self.log_file.split(".")[0] + ".md"
        self.init_msg += f"$ {session_name} will be written to: {self.md_filename}"

        # initialises MD file
        write_string_to_file(
            self.md_filename,
            f"""# {session_name} {begin_time.strftime("%Y/%m/%d %H:%M:%S")}""",
            "w",
        )

    def setup_logger(self):
        # creates output formats for terminal and log file
        formatter1 = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
        formatter2 = logging.Formatter("%(message)s")

        # clears up any existing logger
        my_logger = logging.getLogger()
        my_logger.handlers = []
        for hdlr in my_logger.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                my_logger.removeHandler(hdlr)

        # initializes handlers for terminal and file
        filehandler = logging.FileHandler(self.log_file, "w")
        stream_handler = logging.StreamHandler()

        my_logger.setLevel(logging.INFO)
        filehandler.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)

        my_logger.addHandler(stream_handler)
        my_logger.addHandler(filehandler)

        filehandler.setFormatter(formatter1)
        stream_handler.setFormatter(formatter2)


def print_log(message, status="INFO"):
    """
    Shows message to terminal and logs it to file.
    Compatible with dask workers.
    Used the dask logger that used itself logging logger instance before.

    Parameters
    ----------
    message : str
        message.
    status : str, optional
        either DEBUG, INFO or WARN. The default is 'INFO'.

    Returns
    -------
    None.

    """

    if status == "INFO":
        logger.info(message)
    elif status == "WARN":
        logger.warning(message)
    elif status == "DEBUG":
        logger.debug(message)


def print_framed_text(text: str, frame: str):
    """Example: ================= text =================

    Parameters
    ----------
    text : str
        Text to print in the middle
    frame : str
        Template of frame to put in right and left
        Example: "================="
    """
    nbr_to_remove = len(text) // 2
    dashes = frame[:-nbr_to_remove]
    rest = "" if len(text) % 2 else frame[0]
    print_log(f"{dashes} {text} {dashes}{rest}")


def print_session_name(name: str):
    print_log("")
    print_framed_text(name, "================================")
    print_log("")


def print_dashes():
    print_log("-------------------------------------------------------------------")


def print_title(title: str):
    print_framed_text(title, "--------------------------------")


def print_analyzing_label(text: str):
    print_dashes()
    print_framed_text(text, "                                ")
    print_dashes()


def print_section(section: str):
    print_log(f"$ Load: {section}")


def print_unknown_params(unknown_params: dict):
    print_log(
        f"! Unknown parameters detected: {unknown_params}",
        status="WARN",
    )


def write_string_to_file(file_name, text_to_output, attribute="a"):
    """write a line of text into a file

    Parameters
    ----------
    file_name : str
        log file
    text_to_output : str
        text to write in file
    attribute : str, optional
        Open file mode option, by default "a"
    """
    with open(file_name, mode=attribute, encoding="utf-8") as file_handle:
        file_handle.write(str(text_to_output) + "\n")
