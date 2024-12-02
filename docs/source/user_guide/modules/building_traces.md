# Building chromatin traces



## Pre-processing

Two methods are used for pre-processing localization tables.

- `filter_localizations`: removes localizations that do not achieve a user-defined threshold of intensity.
- `register_localizations`: applies local registration corrections to a localization table.



## Building and processing trace tables

The main method for building chromatin traces is `build_traces`.

Chromatin trace tables can be post-processed using multiple scripts:

- `trace_analyser`: quantifies several quantities from a trace table.
- `trace_filter`: applies simple filtering methods to a trace table.
- `trace_filter_advanced`: applies advanced filtering to trace tables.
- `trace_assign_mask`: assigns a label to traces using a user-provided mask file.
- `trace_merge`: combines/merges multiple trace tables (e.g. from different replicates or FOVs) from anywhere in your file system (more flexible than `trace_combinator`).
- `trace_combinator`: combines/merges multiple trace tables (e.g. from different replicates or FOVs) using the folder architecture of pyHiM.
- `trace_plot`: script to plot one or multiple traces in 3D.



## Building pairwise distance maps

Finally, trace tables can be converted to pairwise distance matrices using either the `pyHiM`  built-in method:`build_matrices`, or using an external script: `trace_to_matrix`).



## Detailed description of methods and scripts

```{toctree}
:maxdepth: 1

building_traces/filter_localizations
building_traces/register_localizations
building_traces/build_traces
building_traces/trace_analyser
building_traces/trace_filter
building_traces/trace_filter_advanced
building_traces/trace_assign_mask
building_traces/trace_merge
building_traces/trace_combinator
building_traces/trace_plot
building_traces/build_matrices
building_traces/trace_to_matrix
```
