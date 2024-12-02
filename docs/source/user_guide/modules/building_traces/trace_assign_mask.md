# trace_assign_mask

Use `trace_assign_mask` to assign specific *labels* to chromatin traces in a trace table.

`trace_assign_mask` will load a trace file and a number of NUMPY-formatted mask files and assign labels. If a trace falls within a mask, then the mask label will be assigned to the corresponding column of the trace table. If a trace falls *at the same time* within multiple masks, multiple labels will be appended to the corresponding column of the trace table. If a trace falls within no mask, then the label column of the trace table will be kept empty.

## Invoke

```bash
$ trace_assign_mask --input trace_file.ecsv --mask_file my_mask.npy --label mymask
```

This will apply the label `mymask` to traces falling within the masks of the file `my_mask.npy`. The output will be a trace file with the extension `labeled`.



Multiple mask files can be provided using piping.

```bash
$ ls my_traces*.ecsv | trace_assign_mask --mask_file my_mask.npy --pipe  --label mymask
```

In this case the `mymask` will be applied to multiple trace files.



## Relevant options

```
usage: trace_assign_mask [-h] [--input INPUT] [--mask_file MASK_FILE] [--pixel_size PIXEL_SIZE] [--label LABEL]
                            [--pipe]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input trace file
  --mask_file MASK_FILE
                        Input mask image file. Expected format: NPY
  --pixel_size PIXEL_SIZE
                        Lateral pixel size un microns. Default = 0.1
  --label LABEL         Label to add to trace file. Default=labeled
  --pipe                inputs Trace file list from stdin (pipe)

```
