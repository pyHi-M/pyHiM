# trace_combinator

*The objective of trace_combinator is to combine traces from different ROIs or different experiments.*

- This script takes JSON file with folders where datasets are stored.
- It searches for Trace files with the expected methods, loads them, and
- combines them into a single table that is outputted to the buildPWDmatrix folder.

## Invoke

`trace_combinator` can be run from the command line in two manners:

- Provide input parameters in a `folders2Load.json` dictionary (see example below) at execution folder or at a `rootFolder` provided as argument. The script will read ALL the `Trace` files in the paths  provided in `folders2Load.json` and will combine them into a unique  `Trace_` files.
  The output of `trace_combinator` will be generated at the `rootFolder` in the directory `combined_traces` that will contain a new `Trace` file with the combined traces.

  <u>*warning*</u>: `trace_combinator` will combine **ALL** traces found in the `buildPWDmatrix` folders within `folders2load.json`. If you want to filter which *Trace* files are used, we recommend you to use the second method described below. Alternatively, you can use the `--method` command-line argument to filter which `Trace` files are used. In that case, make sure to verify the terminal output to ensure the right `Trace` files have been used.

- An alternative way to use `trace_combinator` is to use pipes. In this case, a list of `Trace` files to process is piped into `trace_combinator` and the `--pipe` argument is invoked. Piping can be either done using a command or by sending a list in a file. See examples below.



  Example 1: using `ls`

  ```sh
  # use ls to select which files you want to combine
  $ ls folder1/Trace*3D*6.ecsv
  folder1/Trace_3D_barcode_KDtree_ROI:6.ecsv  folder1/Trace_3D_barcode_mask:DAPI_ROI:6.ecsv  folder1/Trace_3D_barcode_mask:mask0_ROI:6.ecsv

  # then pipe these files into trace_combinator
  $ ls folder1/Trace*3D*6.ecsv | trace_combinator --pipe
  ```

  this will process the three `Trace` files listed using `ls`.



  Example 2: using cat

  ```sh
  # first make list of files to process and write it in a file
  $ cat files_to_combine
  folder1/Trace_3D_barcode_KDtree_ROI:15.ecsv
  folder2/Trace_3D_barcode_KDtree_ROI:6.ecsv

  # then pipe these files into trace_combinator
  $ cat files_to_combine | trace_combinator --pipe
  ```

  this will process the files within `files_to_combine`.



**Use `trace_combinator` to select traces with specific *labels***

`trace_combinator` can also be used to combine traces with specific, user-provided  `labels`, which can be attributed using `trace_selector` (see above). These can be indicated by using the `--label` command-line argument. In addition, you need to indicate whether you want to use only the traces that contain the label (`labeled`), the traces that <u>do not</u> contain the label (`unlabeled`), or all the traces (`all`) irrespective of whether they contain or not the label. It is good practice to verify the `label` column of your output `Trace_` file to check that you selected the expected traces.

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
||||





## Relevant options

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



**folders2Load.json template**

```sh
{
    "dataset_name": {
        "Folders": [
            "/user_disk/data/Experiment_4/0_Embryo/buildsPWDmatrix",
            "/user_disk/data/Experiment_4/1_Embryo/buildsPWDmatrix",
            "/user_disk/data/Experiment_4/2_Embryo/buildsPWDmatrix",
            "/user_disk/data/Experiment_4/4_Embryo/buildsPWDmatrix",
            "/user_disk/data/Experiment_4/5_Embryo/buildsPWDmatrix"
        ],
        "PWD_clim": 1.4,
        "PWD_mode": "median",
        "iPWD_clim": 6,
        "iPWD_mode": "median",
        "ContactProbability_scale": 15,
        "ContactProbability_cmin": 0.0,
        "ContactProbability_distanceThreshold": 0.25
    }
}
```
