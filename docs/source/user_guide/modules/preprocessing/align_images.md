# register_global
*Registers fiducials using a barcode as reference*

## Invoke
Inside the folder with your input data, run:
```shell
pyhim -C register_global
```

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|
|<image_name>.tif|2..n|Yes|2D images with a fiducial channel to align.|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
|register_global.ecsv|1|X, Y shift for each image|

## Relevant options

Parameters for this script will be read from the  ```register_global``` field of ```parameters.json```

|Name|Option|Description|
|:-:|:-:|:-:|
|referenceFiducial| |Selects reference barcode image|
|alignByBlock| | Sets to false if a block correction is not needed. Default: True|


## Description

There are several ways to correct for drift in *pyHiM*:

2.1 **Global drift correction by cross-correlation.** This option just runs a x-correlation between the 2D projected images for the reference and cycle fiducials. It is the fastest, but will ignore local deformations in the sample and, sometimes, can get fooled by bright dirt in the image that will drive the x-correlation to the wrong place. If your sample is clean and does not show much deformation, this is the way to go. This method will output overlapped images that should be used to determine whether the method worked as expected, or whether a local correction is needed.

2.2 **Block drift correlation.** This option will also use the 2D projection images of reference and cycle _fiducials, but it will first break them up into blocks and will perform a block-by-block optimization of XY drift. This method is very robust and is not easily fooled by dirt in the sample. However, this method will find a consensus global drift that will be applied and therefore local drift issues are not solved. An additional advantage of method 1 is that it can estimate how much local drift is present in each block and will use this to discard blocks where the local drift is higher than a user-provided tolerance (see below). After you run this method, you will get the uncorrected and corrected images so you can evaluate whether it worked properly and whether local drift correction methods need to be applied.

2.3 **2D Local drift correction.** This method will be applied after methods 2.1 and 2.2. It will iterate over the DAPI masks detected in the segmentation function (see below), extract a 2D region around each mask, and x-correlate the reference and cycle _fiducials in this 2D sub-region. Thus, this method is slower than methods 1 and 2, but provides local corrections of deformations of the sample. The method will output images with the uncorrected and corrected overlaps for each DAPI mask sub-region so you can evaluate its performance.

## Step by step

In the set of *fiducial* images, one is chosen by initialization parameters to be the reference.
The algorithm takes images one by one and aligns them with the reference.
There are several ways to compute the shift:
- Global alignment makes simple cross-correlation with two images
- Splits image in blocks and makes cross-correlation block by block. The `alignByBlock` parameter in the `alignImages` field of `parameters.json` should be set to `True`. It calculates the optimal shift between fiducial and reference in each block. It estimates the root mean squared error (RMS) between the reference and the shifted image for each block, and uses the blocks in which the RMS is within `tolerance`. Mean and standard deviation of the XY shifts are calculated, and mean shifts are used for shifting the image and getting the final RMS error. This method is more robust against a bright noise spot.
