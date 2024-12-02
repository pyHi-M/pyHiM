# Parameters
*parameters.json file, previously named infoList.json (DEPRECATED)*

## Global structure

Each section in `common` represents a step of pyHiM processing. Parameters are defined by default in `common` in their corresponding sections. If you want to change a parameter just for one channel, go to `labels` and you can overwrite the parameter value in the channel of your choice (an example [here](https://pyhim.readthedocs.io/en/latest/getting_started/tutorials/configuration_file.html#manually)).

## Parameter overview by section
*Parameters are sort by alphabetical order.*

### 1. acquisition

|Parameter name|Used to|
|---|---|
|barcode_channel|List the files to process; label of barcode channel|
|DAPI_channel|List the files to process; label of DAPI channel|
|fiducialBarcode_channel|List the files to process; label of fiducial channel for barcode cycles|
|fiducialDAPI_channel|List the files to process; label of fiducial  channel for the DAPI/RNA cycles|
|fiducialMask_channel|List the files to process; label of fiducial channel for mask cycles|
|fileNameRegExp|Extraction information from filename|
|label|Know which label is currently being processed (added in `convertsParameterFile` method)|
|label_channel|in future this field will contain the ch for the label. This parameter will superseded the individual channel fields above.|
|label_channel_fiducial|in future this field will contain the ch for the label fiducial. This parameter will superseded the individual channel fields above.|
|mask_channel|List the files to process, label of mask channel|
|pixelSizeXY|Get lateral pixel size in nm; compute voxel size for 3D gaussian fitting|
|pixelSizeZ|Get axial pixel size in nm; compute voxel size for 3D gaussian fitting|
|RNA_channel|List the files to process; Label of RNA channel|
|zBinning| Speed up processing time; binning in z-axis. A z-binning of 2 will skip every other plane. A z-binning of 1 will keep all planes.|


### 2. zProject

|Parameter name|Used to|
|---|---|
|blockSize|Split image into blocks for `laplacian` mode|
|display|Save output 2D projection as PNG; show image|
|folder|Give a name of output folder to save output data of zPorject features|
|mode|Select projection mode between `full`, `manual`, `automatic`, `laplacian`|
|windowSecurity|Remove the lowest and highest Z-plans for `automatic` mode|
|zmax|Set a maximum on Z axis for `manual` mode|
|zmin|Set a minimum on Z axis for `manual` mode|
|zProjectOption|Select projection option between sum of all concerned planes (`sum`) or takes the maximum intensity projection (`MIP`)|
|zwindows|Set a margin to find best focal place for both `automatic` and `laplacian` modes|

### 3. alignImages

|Parameter name|Used to|
|---|---|
|alignByBlock|True will perform block alignment. False will do global alignment.|
|background_sigma|Remove inhomogeneous background; set the number of standard deviations to use for both the lower and upper clipping limit ([astropy.stats.SigmaClip](https://docs.astropy.org/en/stable/api/astropy.stats.SigmaClip.html))|
|blockSize|Define size in (X,Y) of block for 3D local alignment; value needs to be a power of 2|
|folder|Give a name of output folder to save output data of register_global features|
|higher_threshold|Set higher threshold to adjust image intensity levels before xcorrelation for `alignment in 2D`|
|localAlignment|Select mode between global alignment (`None`), 2D local alignment (`mask2D`) and 3D local alignment ( `block3D`)|
|lower_threshold|Set lower threshold to adjust image intensity levels before xcorrelation for `alignment in 2D`|
|outputFile|Set a base name for output files|
|referenceFiducial|Set name of the reference fiducial cycle|
|tolerance|Determine the % of error tolerated for `blockAlignment` mode|
|3D_higher_threshold|Set higher threshold to adjust image intensity levels before xcorrelation for `Alignment3D`|
|3D_lower_threshold|Set lower threshold to adjust image intensity levels before xcorrelation for `Alignment3D`|

### 4. segmentedObjects

|Parameter name|Used to|
|---|---|
|area_max|Set max area to keep 2D segmented object (in pixels)|
|area_min|Set min area to keep 2D segmented object (in pixels)|
|background_method|Select method to process background for `barcode`, `DAPI` and `mask` labels, between `flat`, `inhomogeneous` and (just for `DAPI` and `mask`) `stardist` method |
|background_sigma|Remove inhomogeneous background; set the number of standard deviations to use for both the lower and upper clipping limit ([astropy.stats.SigmaClip](https://docs.astropy.org/en/stable/api/astropy.stats.SigmaClip.html))|
|brightest|Set max number of objects segmented per FOV (only for barcodes!); "Number of brightest objects to keep after sorting the full object list. If brightest is set to None, all objects will be selected" ([photutils.DAOStarFinder](https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html))|
|centroidDifference_max|Set max difference between z centroid position determined by moment and by gaussian fitting|
|folder|Give a name of output folder to save output data of segmentedObjects features|
|fwhm|Set source size (in pixels); "The full-width half-maximum (FWHM) of the major axis of the Gaussian kernel in units of pixels." ([photutils.DAOStarFinder](https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html))|
|intensity_max|Set max intensity to keep object|
|intensity_min|Set min intensity to keep object|
|operation|Select 2D or 3D method of segmentation for DAPI and barcode label; Process secondary masks in `process_snd_channel` script for RNA label|
|outputFile|Set a base name for output files|
|residual_max|Set maximum difference between axial spot intensity and gaussian fit|
|sigma_max|Set maximum gaussian fit sigma allowed (axial spot intensity)|
|stardist_basename|Set path name of AI models for 2D segmentation with StarDist|
|stardist_network|Set network name for 2D `DAPI` and `mask` segmentation with StarDist|
|stardist_network3D|Set network name for 3D `barcode`, `DAPI` and `mask` segmentation with StarDist|
|tessellation|Allow tessellation to segment `DAPI` and `mask` if value is `True`|
|threshold_over_std|Set threshold used to detect sources|
|3D_area_max|Set max area to keep 3D segmented object (in pixels)|
|3D_area_min|Set min area to keep 3D segmented object (in pixels)|
|3D_boxSize|Set size of box used for block decomposition|
|3D_contrast|"The fraction of the total (blended) source flux that a local peak must have (at any one of the multi-thresholds) to be considered as a separate object." ([photutils.segmentation.deblend_sources](https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html))|
|3D_lower_threshold|Set lower threshold for adjusting image levels|
|3D_nlevels|"The number of multi-thresholding levels to use." ([photutils.segmentation.deblend_sources](https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html))|
|3D_psf_yx|"Radius of the spot, in nanometer. One value per spatial dimension (zyx or yx dimensions). If it's a scalar, the same radius is applied to every dimensions." ([apifish.detection.spot_modeling.fit_subpixel](https://github.com/apiFISH/apiFISH/blob/development/apifish/detection/spot_modeling.py))|
|3D_psf_z|"Radius of the spot, in nanometer. One value per spatial dimension (zyx or yx dimensions). If it's a scalar, the same radius is applied to every dimensions." ([apifish.detection.spot_modeling.fit_subpixel](https://github.com/apiFISH/apiFISH/blob/development/apifish/detection/spot_modeling.py))|
|3D_sigma|instance a 2D Gaussian filter kernel ([astropy.convolution.Gaussian2DKernel](https://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html))|
|3D_threshold_over_std|Set threshold used to detect 3D sources|
|3D_higher_threshold|Set higher threshold for adjusting image levels|
|3dAP_brightest|Set number of sources sought in each YZ plane; "Number of brightest objects to keep after sorting the full object list. If brightest is set to None, all objects will be selected" ([photutils.DAOStarFinder](https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html))|
|3dAP_distTolerance|Set distance (in pixels) to attribute a source localized in YZ to one localized in XY|
|3dAP_flux_min|Set threshold to keep a source detected in YZ|
|3dAP_window|construct a YZ image by summing from xPlane-window:xPlane+window|
|3DGaussianfitWindow|Set size of window in xy to extract 3D subVolume, in px. 3 means subvolume will be 7x7.|
|3Dmethod|Select which segmentation method to use for the 3D source segmentation between `stardist` or just by thresholding|


### 5. buildsPWDmatrix

|Parameter name|Used to|
|---|---|
|colormaps|Set colormaps used for plotting matrices|
|flux_min|Set minimum flux per spot for 2D. If flux is smaller, localization will be discarded|
|flux_min_3D|Set minimum flux per spot for 3D localizations. If flux is smaller, localization will be discarded|
|folder|Give a name of output folder to save output data of buildsPWDmatrix features|
|KDtree_distance_threshold_mum|Set distance threshold used to build KDtree|
|mask_expansion|Set number of pixels masks will be expanded to assign localizations|
|toleranceDrift|Set tolerance used for block drift correction (in pixels)|
|tracing_method|Set list of methods it will use|
