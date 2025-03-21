# mask_cellpose
This script is provided for the user to be able to segment nuclei or other objects (e.g. mask images) using cellpose.

For more information, follow this tutorial:
[Running cellpose to create 3D masks for pyHiM](../../../getting_started/tutorials/cellpose.md)




## Invoke
To process
```shell
$ mask_cellpose.py --input my_image.tif
```



## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|<image_name>.tif|1|Yes|3D image in tif format.|
|parameters.json|1|Yes|Parameter file.|
|register_global/data/shifts.json|1|Yes|Output JSON file of register_global with the shift values.|


## Outputs
|Name shape|Quantity|Description|
|---|---|---|
|mask_3d/data/*_3Dmasks.npy|1|NUMPY 3D array. The output labeled image.|
|mask_2d/data/*_Masks.npy|1|NUMPY 2D array. The output labeled image, projected by maximum.|
