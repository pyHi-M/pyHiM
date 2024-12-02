# Running cellpose to create 3D masks for pyHiM

*For the 3D mask segmentation, pyHiM use by default a `stardist` model. Follow this tutorial if you want to segment your masks with `cellpose`.*

## Installation

 First, download the [`mask_cellpose.py` script](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/marcnol/pyHiM/blob/development/src/postProcessing/mask_cellpose.py).

full instructions: [GitHub page of cellpose](https://github.com/mouseland/cellpose)

```sh
conda create --name cellpose python=3.8
conda activate cellpose
python -m pip install cellpose
python -m pip install cellpose[gui]
```

## requirements
Before performing the segmentation, you need to apply `project` & `register_global` routines inside the folder containing the images of your masks:
```bash
conda activate pyHiM
pyhim -C project,register_global
```

## Cellpose script for pyHiM

To segment your masks with cellpose in the pyHiM context, you need to run the `mask_cellpose.py` script to:
1. Register your image with the `register_global` shift values.
2. Segment with cellpose.
3. Save segmented masks in the good path and `NPY` format.

### Default segmentation

The parameters are set by default. They were optimized for late embryos but also seem to work well for tissues.

Run this command inside the folder containing the masks with the path of your [downloaded script](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/marcnol/pyHiM/blob/development/src/postProcessing/mask_cellpose.py):
```bash
conda activate cellpose
python3 <path/to/>mask_cellpose.py --input <your_mask_name.tif>
```

The script produce an `NPY` file inside the folder `segmentedObjects/data/`.

```{note}
`segmentedObjects` is the default value of the `mask_3d_folder` parameter.
```

### Personalized segmentation

If you need to optimize new parameters, run the cellpose GUI and note your configuration:

```bash
conda activate cellpose
$ cellpose
```

Cellpose parameters can be changed by providing them as arguments to `mask_cellpose.py`:
```bash
  --cellprob CELLPROB  cellprob threshold. Default = -8.
  --flow FLOW          flow threshold. Default = 10.
  --stitch STITCH      stitch threshold. Default = 0.1.
  --diam DIAM          diameter. Default = 50.
  --model MODEL        pretrained_model to use for running. Default: “cyto”.
```

Example:

```bash
mask_cellpose.py --input scan_001_DAPI_006_ROI_converted_decon_ch00.tif --cellprob -8 --flow 10 --stitch 0.1 --diam 50
```

## CPU or GPU ?

The default mode use the CPU (slower but does not need a GPU!).
The GPU mode can be call by the API or the CLI.
The API works faster but sometimes crashes in small computers with not much memory.
The CLI is more robust to memory requirements.

### CPU

```bash
mask_cellpose.py --input <your_mask_name.tif>
```

### GPU via API

```bash
mask_cellpose.py --input <your_mask_name.tif> --gpu
```

### GPU via CLI

```bash
mask_cellpose.py --input <your_mask_name.tif> --gpu --cli
```

## Continue pyHiM analysis until tracing

After, go back to the pyHiM conda environment and you can run the next routines like this:

```bash
conda activate pyHiM
pyhim -C register_local,localize_3d,filter_localizations,register_localizations,build_traces,build_matrix
```
