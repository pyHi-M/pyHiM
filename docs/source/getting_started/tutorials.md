# Tutorials

## First steps

```{eval-rst}

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 p-2

    ---
    .. image:: tutorials/notebooks/_static/jupyter_logo.png
       :height: 200

    ^^^^^^^^^^

    First steps with Jupyter Notebook

    +++

    .. link-button:: tutorials/run_pyhim_nb
            :type: ref
            :text: Jupyter Notebook
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: ../_static/spot2matrix.png
       :height: 200

    ^^^^^^^^^^

    First steps with 3D analysis of sequential DNA-FISH data

    +++

    .. link-button:: tutorials/notebooks/full_pyHiM_run
            :type: ref
            :text: pyHiM in 3D
            :classes: btn-block btn-info stretched-link text-white
```

## Advanced


```{eval-rst}

.. panels::
    :card: + intro-card text-center
    :column: col-lg-3 col-md-6 col-sm-6 col-xs-12 p-2

    ---
    .. image:: ../_static/getting_started/install.png
       :width: 150

    ^^^^^^^^^^

    Full test in CLI

    +++

    .. link-button:: tutorials/test_installation
            :type: ref
            :text: Test installation
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: ../_static/filetype-json.svg
       :width: 150

    ^^^^^^^^^^^^^^^

    How to customise settings

    +++

    .. link-button:: tutorials/configuration_file
            :type: ref
            :text: Parameters file
            :classes: btn-block btn-info stretched-link text-white


    ---
    .. image:: ../_static/getting_started/logo_cellpose.png

    ^^^^^^^^^^

    Running Cellpose to create 3D masks

    +++

    .. link-button:: tutorials/cellpose
            :type: ref
            :text: Cellpose for pyHiM
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: ../_static/stardist_logo.jpg
       :width: 150

    ^^^^^^^^^^

    Create your model for the segmentation (masks and spots)

    +++

    .. link-button:: tutorials/stardist
            :type: ref
            :text: StarDist for pyHiM
            :classes: btn-block btn-info stretched-link text-white


```

```{toctree}
:maxdepth: 1
:hidden:

Jupyter Notebook<tutorials/run_pyhim_nb>
pyHiM in 3D<tutorials/notebooks/full_pyHiM_run.ipynb>
Test installation<tutorials/test_installation>
tutorials/configuration_file
Cellpose for pyHiM<tutorials/cellpose>
StarDist for pyHiM<tutorials/stardist>
```
