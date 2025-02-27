*********************
pyHiM - Documentation
*********************

**Date**: |today| **Version**: |release|

`pyHiM` implements the analysis of multiplexed DNA-FISH data, as described in our `protocols paper <https://www.nature.com/articles/s41596-019-0269-9>`_.

===============
Getting started
===============

.. panels::
    :card: + intro-card text-center
    :body: text-center p-2
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

    ---
    .. image:: getting_started/tutorials/notebooks/_static/Download-Icon.png
       :height: 180

    +++

    .. link-button:: getting_started/quick_install
            :type: ref
            :text: Installation
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: _static/spot2matrix.png
       :height: 180

    +++

    .. link-button:: getting_started/tutorials/notebooks/full_pyHiM_run
            :type: ref
            :text: 3D pipeline step by step
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: _static/index_getting_started.svg
       :height: 180

    +++

    .. link-button:: getting_started/typical_run
            :type: ref
            :text: Running pyHiM
            :classes: btn-block btn-info stretched-link text-white

=============
Main features
=============

.. panels::
    :card: + intro-card text-center
    :body: text-center p-2
    :footer: text-center p-2
    :column: col-lg-2 col-md-4 col-sm-6 col-xs-12 p-2

    ---
    .. image:: _static/getting_started/dim_space.gif
       :height: 100


    +++

    .. link-button:: user_guide/modules/preprocessing/make_projections
            :type: ref
            :text: 2D or 3D Pipeline
            :classes: stretched-link

    ---
    .. image:: _static/registration_zoom.svg
       :height: 100

    +++

    .. link-button:: user_guide/modules/preprocessing/align_images
            :type: ref
            :text: Image registration
            :classes: stretched-link

    ---
    .. image:: _static/segmentation_mask.png
       :width: 100

    +++

    .. link-button:: user_guide/modules/identification/segment_masks_3d
            :type: ref
            :text: Segment nuclei
            :classes: stretched-link


    ---
    .. image:: _static/localizations.svg
       :width: 100

    +++

    .. link-button:: user_guide/modules/identification/segment_sources_3d
            :type: ref
            :text: Localize spots
            :classes: stretched-link

    ---
    .. image:: _static/trace_zoom.png
       :height: 90

    +++

    .. link-button:: user_guide/modules/building_traces/build_traces
            :type: ref
            :text: Chromatin traces
            :classes: stretched-link

    ---
    .. image:: _static/matrix_example.png
       :width: 100

    +++

    .. link-button:: user_guide/modules/building_traces/build_matrices
            :type: ref
            :text: Distance maps
            :classes: stretched-link

==============
Documentation
==============


.. panels::
    :card: + intro-card text-center
    :column: col-lg-4 col-md-6 col-sm-6 col-xs-12 p-2

    ---
    .. image:: _static/index_user_guide.svg
       :width: 60

    User guide
    ^^^^^^^^^^

    Find in-depth information on the
    key concepts of *pyHiM* with background information and explanation.

    +++

    .. link-button:: user_guide/pyhim_presentation
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: _static/index_api.svg
       :width: 56

    API reference
    ^^^^^^^^^^^^^

    The reference guide describes the *pyHiM* API and how the methods work with which parameters.

    +++

    .. link-button:: reference/pyhim_class_diagram
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-info stretched-link text-white

    ---
    .. image:: _static/index_contribute.svg
       :width: 50

    Developer guide
    ^^^^^^^^^^^^^^^

    The contributing guidelines will guide
    you through the process of improving *pyHiM*.

    +++

    .. link-button:: contributor/dev_env
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-info stretched-link text-white


`pyHiM` is a software package developed by the `Nollmann Lab <http://www.nollmannlab.org>`_ at the `Center of Structural Biology <http://www.cbs.cnrs.fr>`_, a department of the `CNRS <http://www.cnrs.fr>`_ and the `INSERM <http://www.inserm.fr>`_.


.. toctree::
   :caption: Getting Started
   :hidden:

   getting_started/quick_install
   getting_started/tutorials
   getting_started/typical_run

.. toctree::
   :caption: User Guide
   :hidden:

   user_guide/pyhim_presentation
   user_guide/pipeline_overview
   user_guide/fundamental

.. toctree::
   :caption: Reference
   :hidden:

   reference/Chromatin_trace_table
   reference/pyhim_class_diagram
   reference/infoList_comprehension
   reference/apidoc/modules

.. toctree::
   :caption: Contributor's Guide
   :hidden:

   contributor/dev_env
   contributor/dev_installation
   contributor/coding_style
   contributor/dev_process
   contributor/good_commit
   contributor/build_doc
   contributor/no_python_files
   contributor/issues
   contributor/work_with_apifish
   contributor/resources
   contributor/license
