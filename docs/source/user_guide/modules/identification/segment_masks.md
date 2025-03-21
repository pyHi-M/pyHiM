# mask_2d and localize_2d
*Segments DAPI and sources in 2D*

## Invoke
Inside the folder with your input data, run:
```shell
pyhim -C mask_2d,localize_2d
```

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|
|<image_name>.tif|2..n|Yes|2D images|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
|segmented_barecode.ecsv|1|If it's barecode object|
|segmented_mask.npy|2..n|If it's mask object|


## Relevant options
|Name|Option|Description|
|:-:|:-:|:-:|
|operation|2D| Select 2D mask segmentation|
||3D| Select 3D mask segmentation|
|background_method|inhomogeneous| |
||flat| |
||stardist| |
|stardist_network| | Name of the network used for segmentation|
|stardist_basename| | Folder containing AI models|
|background_sigma| | Used to remove inhomogeneous background. Default: 3.0|
|threshold_over_std| | Threshold used to detect sources. Default: 1.0|
|area_min| | Minimal area to keep object|
|area_max| | Maximal area to keep object|
|residual_max| | Maximum difference between axial spot intensity and gaussian fit|


## Description
A 2D mask segmentation produces two outputs saved in the `segmentedObjects` folder:

```
scan_002_mask0_002_ROI_converted_decon_ch01_segmentedMasks.png
scan_002_mask0_002_ROI_converted_decon_ch01_Masks.npy
```

The PNG file is a representation of the raw image and the segmented objects.

The NPY file is a 2D labeled numpy array containing the segmented objects with an identical size to the original image. Background has a value of _0_ and each mask contains a different integer. The maximum value in this matrix corresponds to the number of masks detected. The file name is constructed using the original root filename with the tag `_Masks`.

_Warning_: This mode operates in 2D, therefore the Startdist network provided **must be** in 2D.
