# trace_filter_advanced

This script is intended to filter trace files.

Multiple filtering options are available:

- spatial filtering: user provided x-y-z limits are provided and localizations outside these limits are removed (--xmin, --xmax, --ymin, --ymax, --zmin, --zmax).
- repeated barcodes: spots from the same cycle appearing in the same trace are removed (--clean_spots).
- minimum number of barcodes: only traces with a user-defined minimum number of barcodes are kept (--N_barcodes).
- barcode removal: user-defined barcodes are defined from all traces (--remove_barcode).
- labeling: traces containing a user-defined label (column in a trace file) are either kept (--label) or removed (--remove_label).

 and takes as input the minimum number of spots a trace needs to be kept.

How to use it:

- Example 1: Invoke from `rootFolder`

  ```sh
  # Call trace_filter from rootFolder to process all Trace files in `buildPWDmatrix`
  $ trace_filter -- N_barcodes 2
  ```

  this will process all  `Trace` files in `buildPWDmatrix`



- Example 2: using cat or ls to provide a specific list of Trace files to process

  ```sh
  # either make list of files to process and write it in a file
  $ cat files_to_combine
  folder1/Trace_3D_barcode_KDtree_ROI:15.ecsv
  folder2/Trace_3D_barcode_KDtree_ROI:6.ecsv

  # then pipe these files into trace_combinator
  $ cat files_to_combine | trace_combinator --N_barcodes 2 --pipe

  # OR use `ls` to select which files you want to combine
  $ ls folder1/Trace*3D*6.ecsv
  folder1/Trace_3D_barcode_KDtree_ROI:6.ecsv  folder1/Trace_3D_barcode_mask:DAPI_ROI:6.ecsv  folder1/Trace_3D_barcode_mask:mask0_ROI:6.ecsv

  # then pipe these files into trace_combinator
  $ ls folder1/Trace*3D*6.ecsv | trace_combinator --N_barcodes 2 --pipe
  ```



**Relevant options**

```sh
usage: trace_filter [-h] [-F ROOTFOLDER] [--N_barcodes N_BARCODES] [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  -O OUTPUT, --output OUTPUT
                        Tag to add to the output file. Default = filtered
  --pipe                inputs Trace file list from stdin (pipe)
  --clean_spots         remove barcode spots repeated in a single trace
  --remove_label        Use this argument to remove traces with the label provided
  --input INPUT         Name of input trace file.
  --N_barcodes N_BARCODES
                        minimum_number_barcodes. Default = 2
  --dist_max DIST_MAX   Maximum distance threshold. Default = np.inf
  --z_min Z_MIN         Z minimum for a localization. Default = 0
  --z_max Z_MAX         Z maximum for a localization. Default = np.inf
  --y_min Y_MIN         Y minimum for a localization. Default = 0
  --y_max Y_MAX         Y maximum for a localization. Default = np.inf
  --x_min X_MIN         X minimum for a localization. Default = 0
  --x_max X_MAX         X maximum for a localization. Default = np.inf
  --remove_barcode REMOVE_BARCODE
                        name of barcode to remove
  --label LABEL         Select traces containing this label, removes all other traces.

```
