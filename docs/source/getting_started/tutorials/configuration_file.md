# Parameters file

*All pyHiM parameters are gathered in a single configuration file called `parameters.json`.*

This file can be edited [manually](#manually) or using a [graphical user interface](#graphical-user-interface).

You can find a global description of each parameter in the [reference guide](../../reference/infoList_comprehension.md).

## Manually

- Copy an `parameters.json` file in the folder where you want to run `pyHiM`.

```{note}
A typical example can be find on [GitHub <img src="../../_static/getting_started/Download-Icon.png" width="50"/>](https://github.com/marcnol/pyHiM/blob/development/src/toolbox/parameter_file/parameters.json)
```

- With a text editor, update the relevant parameters. For example, the  name of the `referenceFiducial` needs to be changed according to your experiment settings.

- The `common` section defines the default parameters for each label. For each label, you can personalize the value of a parameter by  indicating its new value in the `label` section.

**Example:** The Z projection will be done by Maximum Intensity Projection (`MIP`) for all labels except for the barcode images, which will use the `sum` method:

```json
    "common": {
        "zProject": {
            "zProjectOption": "MIP",
            "..."
        },
        "..."
    },
    "labels": {
        "barcode": {
            "zProject": {
                "zProjectOption": "sum",
            }
        },
        "..."
    }
```

## Graphical user interface

You can also create and modify `parameters.json` with an interface. In the folder where your data are saved, launch the editor:

```sh
pyhim_parameters
```

Indicate your parameters and click on `Save settings`. An `parameters.json` file will be created inside your folder with the modified parameters.

```{note}
If you have modified an existing `parameters.json`, a copy of the previous version will be saved in the file named `parameters_preversion.json`.
```

![Screenshot of "standard settings" window](../../_static/standard_settings.png)
