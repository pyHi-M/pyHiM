# trace_merge

This scripts implements a simpler version of trace_combinator, the objective is to merge different trace files coming from different FOVs or experiments. It is more flexible than trace_combinator in that it just gets the input file names from a pipe and merges them into a user provided (or default) output trace file.

## Invoke

This just takes a list of trace files and merges them together

```
$ ls Trace*.ecsv | trace_merge
```

outputs

ChromatinTraceTable() object and output .ecsv formatted file with assembled trace tables.

```
usage: trace_combinator [-h] [-F ROOTFOLDER] [-P PARAMETERS] [-A LABEL]
                           [-W ACTION] [--saveMatrix] [--ndims NDIMS]
                           [--method METHOD] [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  -F ROOTFOLDER, --rootFolder ROOTFOLDER
                        Folder with images
  -P PARAMETERS, --parameters PARAMETERS
                        Provide name of parameter files. folders2Load.json
                        assumed as default
  -A LABEL, --label LABEL
                        Add name of label (e.g. doc)
  -W ACTION, --action ACTION
                        Select: [all], [labeled] or [unlabeled] cells plotted
  --saveMatrix          Use to load matlab formatted data
  --ndims NDIMS         Dimensions of trace
  --method METHOD       Method or mask ID used for tracing: KDtree, mask, DAPI
  --pipe                inputs Trace file list from stdin (pipe)
```
