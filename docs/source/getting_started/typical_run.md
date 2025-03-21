# Running pyHiM

## Prepare run

- Copy the TIFF images you want to process into a single folder.
  ```{note}
  This folder will be called the `input_directory`
  ```

- Copy or create a file named `parameters.json` into your `input_directory`.

  ```{note}
   This file contains all the input parameters required to run `pyHiM`. [You can download and unzip an example here](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/marcnol/pyHiM/blob/development/src/toolbox/parameter_file/parameters.json).
  ```

- Update the `parameters.json` file to indicate the **reference cycle** used for drift correction.
  ```{note}
  This can be done by manually editing the `parameters.json` file or by running the graphical user interface provided in the script: `pyhim_parameters` ([tutorial here](tutorials/configuration_file.md)).
  ```

## Basic run

- Open a terminal (Linux) or a Anaconda Prompt (windows)

- Move to the input_directory :

  ```bash
	cd input_directory
	```

- Activate your conda environment:

  ```bash
	conda activate pyHiM
	```

- To run the basic pyHiM pipeline :
	```bash
	pyhim
	```

  ```{note}
  The basic *pyHiM* pipeline will:
  1. Project 3D images
  2. Calculate drift based on fiducial images
  3. Correct drift for nuclei and barcode images (i.e. registration)
  4. Segment and localize barcode spots
  5. Segment nuclei and masks
  6. Match spots to masks to build chromatin traces
  7. Build single-trace and ensemble pairwise distance maps
  ```

- Each module will create and store its results in a single dedicated folder inside the `input_directory`. A summary of all outputs will also be provided as a markdown file `HiM_Analysis<DDMMYYYY_HHMMSS>.md` saved in the `input_directory`.

## Optional arguments

If you require help, you can call `pyHiM` with the help option :
```bash
pyhim -h
```

The output should look as follows:

```sh
usage: pyhim [-h] [-C CMD] [-F ROOTFOLDER] [-P PARAMETERS] [-T THREADS]

optional arguments:
  -h, --help            show this help message and exit
  -C CMD, --cmd CMD     Comma-separated list of routines to run. DEFAULT:
                        project,register_global,register_local,mask_2d,
                        localize_2d,mask_3d,localize_3d,filter_localizations,
                        register_localizations,build_traces,build_matrix
  -F ROOTFOLDER, --rootFolder ROOTFOLDER
                        Folder path with input images. DEFAULT: Current
                        directory
  -P PARAMETERS, --parameters PARAMETERS
                        Path of the parameters.json file. DEFAULT: pyHiM
                        expect this file inside your current directory
  -T THREADS, --threads THREADS
                        Thread number to run with parallel mode. DEFAULT: 1
                        (sequential mode)
```



- ```-F ``` or ```--rootFolder``` indicates the rootFolder where *pyHiM* expects to find the dataset.

- ```-C or --cmd``` is an optional argument that can be used to run a specific set of functions detailed as a **comma separated list without space**.
  ```{note}
  The mode of action will be determined from the `parameters.json` [configuration file](tutorials/configuration_file.md).
  ```

- ```--threads``` will ask *pyHiM* to run in parallel using multiple threads in your computer or computer cluster.
  ```{note}
  To visualize the progress of your run,  open your browser in `http://localhost:8787` and make sure you connect by

    `ssh -L 8787:localhost:8787 username@servername`
  ```

## Running in multiple folders
To run `pyHiM` in multiple folders we recommend that you create a **BASH script** and provide the location of each `input_directory` that needs to be processed. For this, run `pyHiM` with this way:
```sh
pyhim -F <input_directory>
```
Where `<input_directory>` is the relative or absolute path to the folder containing your input data.
