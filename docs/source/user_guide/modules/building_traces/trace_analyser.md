# trace_analyser

*This script in /src/postProcessing/trace_analyzer.py will read a trace table and derive several different analysis, including the distribution of number barcodes per trace: total, duplicated, and non-duplicated. It also exports the Rg distribution.*

## Invoke


To analyse a single or a list of unrelated files:

```
$ ls buildsPWDmatrix/Trace_2D_barcode_mask:mask0_ROI:2.ecsv | trace_analyzer --pipe
```

To analyze all trace files in `buildPWDmatrix`:

```
$  trace_analyzer
```

Example:

```
$ trace_analyzer --input Trace_3D_barcode_mask:DAPI_ROI:1_filtered.ecsv
```

will analyze the trace file and produce several outputs, including:


![Trace_3D_barcode_mask:DAPI_ROI:3_filtered_traces_XYZ_ROI3](https://github.com/marcnol/pyHiM/assets/341757/2b3f32f2-d9a6-41c8-98b7-372cc60a0439)

![Trace_3D_barcode_mask:DAPI_ROI:3_filtered_trace_statistics](https://github.com/marcnol/pyHiM/assets/341757/281cf895-d043-422c-a7c1-5fc7dcbbf857)

![Trace_3D_barcode_mask:DAPI_ROI:3_filtered_xyz_statistics](https://github.com/marcnol/pyHiM/assets/341757/c33e6a4b-0678-4f1e-b1cd-8834e0779560)
