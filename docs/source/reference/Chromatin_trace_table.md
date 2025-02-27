# pyHiM chromatin trace format

This page describes the format of pyHiM chromatin trace tables.

## Summary

The chromatin trace table is the main output of pyHiM and records the primary results. The table contains the 3D coordinates of single DNA-FISH spots that are spatially linked together in single 3D `chromatin traces`.

Tables have to have 12 columns:
- {name: Spot_ID, datatype: string}
- {name: Trace_ID, datatype: string}
- {name: x, datatype: float32}
- {name: y, datatype: float32}
- {name: z, datatype: float32}
- {name: Chrom, datatype: string}
- {name: Chrom_Start, datatype: int64}
- {name: Chrom_End, datatype: int64}
- {name: 'ROI #', datatype: int64}
- {name: Mask_id, datatype: int64}
- {name: 'Barcode #', datatype: int64}
- {name: label, datatype: string}

`Spot_ID` contains a unique identifier for a spot. Spots with the same `Trace_ID`, by definition, belong to the same chromatin trace. `xyz` coordinates are provided in $\um$ (microns) units. `Chrom` describes the identity of the chromosome being targeted, with `Chrom_Start` and `Chrom_End` containing the genomic coordinates in basepairs. `ROI #` describes the Region of Interest where the trace was acquired, `Mask_id` describes the identity of the mask to which the trace was associated with, `Barcode #` indicates the identity of the barcode, and `label` the name of the region or cell type associated with the trace.

In this table the reported x, y, z coordinates are assumed to result from post-processing and quality control procedures and therefore correspond to the final localization of the DNA target under study.

Trace table have extension `ecsv` and are written and read by astropy or by pandas.

## Example

The following contains the first rows of a typical chromatin trace table.

```bash
# %ECSV 1.0
# ---
# datatype:
# - {name: Spot_ID, datatype: string}
# - {name: Trace_ID, datatype: string}
# - {name: x, datatype: float32}
# - {name: y, datatype: float32}
# - {name: z, datatype: float32}
# - {name: Chrom, datatype: string}
# - {name: Chrom_Start, datatype: int64}
# - {name: Chrom_End, datatype: int64}
# - {name: 'ROI #', datatype: int64}
# - {name: Mask_id, datatype: int64}
# - {name: 'Barcode #', datatype: int64}
# - {name: label, datatype: string}
# schema: astropy-2.0
Spot_ID Trace_ID x y z Chrom Chrom_Start Chrom_End "ROI #" Mask_id "Barcode #" label
c001209e-0a70-4d72-b0d9-1541747a48bb 9d75fd7d-b170-4703-bd2c-810e77757990 100.532814 10.439153 10.996869 xxxxx 0 999999999 17 12 3 OK107
ae281ab3-1c01-4e1b-8dff-c6fa7242548b 9d75fd7d-b170-4703-bd2c-810e77757990 100.79655 10.506308 11.150922 xxxxx 0 999999999 17 12 12 OK107
3eafffb3-0d84-4528-b689-3babbe7e6f25 9d75fd7d-b170-4703-bd2c-810e77757990 100.55611 10.479042 11.213325 xxxxx 0 999999999 17 12 10 OK107
59a53a22-9f97-4253-a2b7-85c7bf388d5f 9d75fd7d-b170-4703-bd2c-810e77757990 100.588234 10.45398 11.16187 xxxxx 0 999999999 17 12 708 OK107
eeb8eaff-cb9a-4b6d-be7c-dd1fdf7ed903 9d75fd7d-b170-4703-bd2c-810e77757990 100.54249 10.473159 11.117058 xxxxx 0 999999999 17 12 8 OK107
f5898628-5dc9-475f-ae8a-50a585442c7b 9d75fd7d-b170-4703-bd2c-810e77757990 100.62601 10.43174 11.196201 xxxxx 0 999999999 17 12 639 OK107
c69259b8-fe30-461f-b254-948b0fb39eee 9d75fd7d-b170-4703-bd2c-810e77757990 100.50262 10.438758 11.308915 xxxxx 0 999999999 17 12 11 OK107
a706a795-3b16-47d8-a79c-6ca89645b5a8 9d75fd7d-b170-4703-bd2c-810e77757990 100.492516 10.462756 11.23178 xxxxx 0 999999999 17 12 9 OK107
2e2ba97a-c56c-42d5-87a7-38d1a44850a5 f869e519-4055-4b33-a908-baf2b832db2f 86.96675 166.51755 9.938585 xxxxx 0 999999999 17 17 17 OK107
1ec18364-f716-4cd0-aa5c-d8a31d99ce83 f869e519-4055-4b33-a908-baf2b832db2f 86.920494 166.58627 9.994431 xxxxx 0 999999999 17 17 16 OK107
49934fc9-be19-4720-97b8-aef2ee64b904 2b53ea53-b85d-4012-a391-df4db9af3e4b 107.340935 68.58549 9.746644 xxxxx 0 999999999 17 36 20 OK107
d0756e97-8ebc-4e6d-9826-e868f52fc01b 2b53ea53-b85d-4012-a391-df4db9af3e4b 107.22091 68.564255 9.5504875 xxxxx 0 999999999 17 36 708 OK107
d106fc7a-2ba1-4ff1-8064-07604616b5f1 2b53ea53-b85d-4012-a391-df4db9af3e4b 107.28838 68.52376 9.712829 xxxxx 0 999999999 17 36 21 OK107

```
