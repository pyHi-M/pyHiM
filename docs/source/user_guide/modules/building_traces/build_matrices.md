# build_matrices

*This script will build single-cell pair-wise distance (PWD) matrices, proximity frequency maps, and N-matrices from each `Trace_` file in the `buildsPWDmatrix` folder. *

## Invoke
Inside the folder with your input data, run:
```shell
pyhim -C build_matrix
```

![build_matrix](../../../_static/from_tuto/build_matrix.png)

## Inputs

|Name shape|Quantity|Mandatory|Description|
|---|---|---|---|
|parameters.json|1|Yes|Parameter file.|

## Outputs
|Name shape|Quantity|Description|
|---|---|---|
||||

## Relevant options

Parameters to run this script will be read from the ```buildsPWDmatrix``` field of ```parameters.json```.

```
"colormaps":{"PWD_KDE":"terrain","PWD_median":"terrain","contact":"coolwarm","Nmatrix":"Blues"},
```


Output images:

- `_PWDhistograms.png`
- `_Nmatrix.png`
- `HiMmatrix.png`
- `_PWDmatrixMedian.png`
- `_PWDmatrixKDE.png`

*example uses fiducial mask*

| method    | contact matrices                                             | **PWD matrix**                                               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2D - mask | ![image-20220212093032574](../../../_static/user_guide/image-20220212093032574.png) | ![image-20220212093119700](../../../_static/user_guide/image-20220212093119700.png) |
| 3D - mask | ![image-20220212093245315](../../../_static/user_guide/image-20220212093245315.png) | ![image-20220212093210913](../../../_static/user_guide/image-20220212093210913.png) |
| KDtree 3D | ![image-20220213120843091](../../../_static/user_guide/image-20220213120843091.png) | ![image-20220213120807698](../../../_static/user_guide/image-20220213120807698.png) |
| Nmatrices | Masking![image-20220212093324905](../../../_static/user_guide/image-20220212093324905.png) | KDTREE![image-20220213120921749](../../../_static/user_guide/image-20220213120921749.png) |
