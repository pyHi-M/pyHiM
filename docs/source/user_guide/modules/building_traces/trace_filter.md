# trace_filter

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
  $ trace_filter --input my_trace.ecsv --N_barcodes 5
  ```

  This will remove traces with less than 5 barcodes.



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
