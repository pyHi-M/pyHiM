# Home made stardist model

You can find instructions to train your own stardist model in different places:
- [StarDist GitHub page](https://github.com/stardist/stardist)
- [ZeroCostDL4Mic Jupyter Notebooks for StarDist 2D](https://github.com/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_2D_ZeroCostDL4Mic.ipynb)
- [ZeroCostDL4Mic Jupyter Notebooks for StarDist 3D](https://github.com/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_3D_ZeroCostDL4Mic.ipynb)


The output structure of your model would look like this:
```bash
/home/user
    └──model_folder
       ├── PSF_2D_model
       │   ├── config.json
       │   ├── thresholds.json
       │   ├── weights_best.h5
       │   └── weights_last.h5
       └── PSF_3D_model
           ├── config.json
           ├── thresholds.json
           ├── weights_best.h5
           └── weights_last.h5
```

To run pyHiM with your models, you need to add `model_folder` path and the network name inside the `parameters.json` file:

```json
{
    "common": {

    },
    "labels": {
        "barcode": {
            "segmentedObjects": {

                "stardist_basename": "/home/user/model_folder",
                "stardist_network": "PSF_2D_model",
                "stardist_network3D": "PSF_3D_model"

            }
        }
    }
}
```

With this example, pyHiM will use your stardist models during the `localize_2d` and `localize_3d` routines.

But for the DAPI segmentation (`mask_2d` and `mask_3d`), pyHiM will always use the default built-in model.
