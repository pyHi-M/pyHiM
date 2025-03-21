#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for file management
"""

import multiprocessing
import os

import numpy as np
from dask.distributed import Client, LocalCluster, get_client

from core.pyhim_logging import print_log


class DaskCluster:
    """Used to manage parallel run thanks the Dask package"""

    def __init__(self, requested_nb_nodes, maximum_load=0.6, memory_per_worker=12000):
        self.requested_nb_nodes = requested_nb_nodes
        # self.n_threads will be created after exetution of initialize_cluster()
        self.n_threads = None
        self.maximum_load = maximum_load  # max number of workers that I can take
        self.memory_per_worker = memory_per_worker  # in Mb
        self.initialize_cluster()
        self.cluster = None
        self.client = None

    def initialize_cluster(self):
        """Defines the number of threads allocated"""
        number_cores_available = multiprocessing.cpu_count()

        # we want at least 12 GB per worker
        free_m = int(os.popen("free -t -m").readlines()[1].split()[-1])

        max_number_threads = int(
            np.min(
                [
                    number_cores_available * self.maximum_load,
                    free_m / self.memory_per_worker,
                ]
            )
        )

        self.n_threads = int(np.min([max_number_threads, self.requested_nb_nodes]))

        print_log(
            f"$ Cluster with {self.n_threads} workers started ({self.requested_nb_nodes} requested)"
        )

    def create_distributed_client(self):
        """Instance workers"""
        client = try_get_client()
        if client is not None:
            print_log("# Shutting down existing cluster! ")
            client.shutdown()
        else:
            print_log("$ No running cluster detected. Will start one.")

        self.cluster = LocalCluster(
            n_workers=self.n_threads,
            threads_per_worker=1,
            memory_limit="64GB",
        )
        self.client = Client(self.cluster)

        print_log("$ Go to http://localhost:8787/status for information on progress...")


def try_get_client():
    """Check if client is alive

    Returns
    -------
    Dask.Client
        Client instance or None
    """
    try:
        client = get_client()
        client.restart()
    except ValueError:
        client = None

    return client
