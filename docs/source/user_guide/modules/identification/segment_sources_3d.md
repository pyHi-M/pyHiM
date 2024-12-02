# localize_3d

*Segments sources in 3D*

## Invoke

Inside the folder with your input data, run:
```shell
pyhim -C localize_3d
```

![localization](../../../_static/from_tuto/localization.png)

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|
|<image_name>.tif|2..n|Yes|3D images|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
|3D_segmented_barcode.ecsv|1||
