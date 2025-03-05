#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os
import platform

# import sys
import time

# import dask.array as da
# import dask.distributed
import numpy as np
from scipy import ndimage
from skimage import io

# from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation

# from tqdm import tqdm  # For progress tracking

try:
    import SimpleITK as sitk

    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print(
        "Warning: SimpleITK not found. Install with 'pip install SimpleITK' for B-spline registration."
    )

try:
    import cupy as cp

    GPU_AVAILABLE = True
    a = cp.ones(1)
    print(a)
except ImportError:
    GPU_AVAILABLE = False

from typing import Any, Dict, Optional, Tuple


def load_tiff_image(filepath: str) -> np.ndarray:
    """
    Load a TIFF image using skimage

    Parameters:
    -----------
    filepath : str
        Path to the TIFF image file

    Returns:
    --------
    ndarray: Loaded 3D image
    """
    try:
        image = io.imread(filepath)
        # Ensure 3D format
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 4:
            image = image.squeeze()
        return image
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        raise


def save_tiff_image(image: np.ndarray, filepath: str):
    """
    Save a 3D image as TIFF

    Parameters:
    -----------
    image : ndarray
        3D image to save
    filepath : str
        Destination file path
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        io.imsave(filepath, image.astype(np.float32))
    except Exception as e:
        print(f"Error saving image {filepath}: {e}")
        raise


def detect_hardware():
    """
    Detect computational hardware and print information

    Returns:
    --------
    str: Hardware type ('CPU' or 'GPU')
    """
    # Check for GPU
    if GPU_AVAILABLE:
        print("GPU Acceleration: Available")
        return "GPU"

    # CPU Information
    cpu_count = multiprocessing.cpu_count()
    cpu_name = platform.processor()
    print(f"CPU Acceleration: {cpu_count} cores")
    print(f"CPU: {cpu_name}")
    return "CPU"


def global_3d_registration(
    reference_image: np.ndarray, target_image: np.ndarray, upsample_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform global 3D registration to correct for large-scale drift

    Parameters:
    -----------
    reference_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image to register
    upsample_factor : int, optional
        Subpixel registration precision

    Returns:
    --------
    registered_image : ndarray
        Globally registered image
    shift : ndarray
        Global 3D shift vector (z, y, x)
    """
    print("Performing global 3D registration...")
    start_time = time.time()

    # Normalize images for comparison
    ref_norm = (reference_image - reference_image.mean()) / (
        reference_image.std() + 1e-8
    )
    target_norm = (target_image - target_image.mean()) / (target_image.std() + 1e-8)

    # Calculate 3D shift using phase correlation
    shift, _, _ = phase_cross_correlation(
        ref_norm, target_norm, upsample_factor=upsample_factor
    )

    # Apply shift to get globally aligned image
    registered_image = ndimage.shift(target_image, shift, mode="nearest")

    print(f"Global registration complete in {time.time() - start_time:.2f} seconds")
    print(f"Global shift: {shift}")

    return registered_image, shift


def bspline_registration_3d(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    grid_size: Tuple[int, int, int] = (5, 5, 3),
    iterations: int = 100,
    sampling_percentage: float = 0.2,
) -> Tuple[np.ndarray, Any]:
    """
    Perform B-spline based deformable registration

    Parameters:
    -----------
    reference_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image to register
    grid_size : tuple
        Control point grid size for B-spline transform
    iterations : int
        Maximum number of iterations for optimizer
    sampling_percentage : float
        Percentage of voxels to sample for metric evaluation

    Returns:
    --------
    registered_image : ndarray
        Deformably registered image
    transform : sitk.Transform
        B-spline transform
    """
    if not SITK_AVAILABLE:
        raise ImportError("SimpleITK is required for B-spline registration")

    print(f"Performing B-spline registration with grid size {grid_size}...")
    start_time = time.time()

    # Convert numpy arrays to SimpleITK images
    reference_sitk = sitk.GetImageFromArray(reference_image.astype(np.float32))
    target_sitk = sitk.GetImageFromArray(target_image.astype(np.float32))

    # Initialize registration method
    registration = sitk.ImageRegistrationMethod()

    # Use mutual information as similarity metric
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)

    # Multi-resolution framework for efficiency
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])

    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup B-spline transform
    transform_domain_mesh_size = grid_size

    """
    transform_domain_physical_dimensions = [
        reference_sitk.GetWidth(),
        reference_sitk.GetHeight(),
        reference_sitk.GetDepth(),
    ]
    transform_domain_origin = [0.0, 0.0, 0.0]
    transform_domain_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    """

    # Create B-spline transform
    bspline_transform = sitk.BSplineTransformInitializer(
        reference_sitk, transform_domain_mesh_size
    )

    # Set initial transform
    registration.SetInitialTransform(bspline_transform)

    # Execute registration
    print("Starting B-spline optimization...")
    final_transform = registration.Execute(reference_sitk, target_sitk)

    # Apply transform to the target image
    registered_sitk = sitk.Resample(
        target_sitk,
        reference_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        target_sitk.GetPixelID(),
    )

    # Convert back to numpy array
    registered_image = sitk.GetArrayFromImage(registered_sitk)

    print(f"B-spline registration complete in {time.time() - start_time:.2f} seconds")

    return registered_image, final_transform


def advanced_3d_registration(
    reference_filepath: str,
    target_filepath: str,
    output_filepath: Optional[str] = None,
    global_upsample_factor: int = 10,
    bspline_grid_size: Tuple[int, int, int] = (5, 5, 3),
    bspline_iterations: int = 100,
    output_transform_filepath: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Comprehensive registration workflow combining global alignment and B-spline deformation

    Parameters:
    -----------
    reference_filepath : str
        Path to reference TIFF image
    target_filepath : str
        Path to target TIFF image to register
    output_filepath : str, optional
        Path to save registered image
    global_upsample_factor : int
        Upsample factor for global registration
    bspline_grid_size : tuple
        Control point grid density for B-spline registration
    bspline_iterations : int
        Number of iterations for B-spline optimization
    output_transform_filepath : str, optional
        Path to save transformation parameters

    Returns:
    --------
    registered_image : ndarray
        Fully registered image
    registration_info : dict
        Dictionary containing registration parameters and results
    """
    # Start overall timing
    total_start_time = time.time()

    # Detect hardware
    hardware = detect_hardware()

    # Load images with timing
    print("Loading images...")
    load_start_time = time.time()
    reference_image = load_tiff_image(reference_filepath)
    target_image = load_tiff_image(target_filepath)
    load_time = time.time() - load_start_time
    print(f"Image loading time: {load_time:.2f} seconds")
    print(
        f"Image shapes: Reference {reference_image.shape}, Target {target_image.shape}"
    )

    # Step 1: Global 3D registration for large-scale drift
    print("\n--- STEP 1: GLOBAL REGISTRATION ---")
    globally_registered, global_shift = global_3d_registration(
        reference_image, target_image, upsample_factor=global_upsample_factor
    )

    # Step 2: B-spline deformable registration for local deformations
    print("\n--- STEP 2: B-SPLINE DEFORMATION ---")
    try:
        fully_registered, bspline_transform = bspline_registration_3d(
            reference_image,
            globally_registered,
            grid_size=bspline_grid_size,
            iterations=bspline_iterations,
        )
        bspline_applied = True
    except Exception as e:
        print(f"B-spline registration failed: {e}")
        print("Using globally registered image instead.")
        fully_registered = globally_registered
        bspline_transform = None
        bspline_applied = False

    # Save registered image with timing
    if output_filepath:
        print("\nSaving registered image...")
        save_start_time = time.time()
        save_tiff_image(fully_registered, output_filepath)
        save_time = time.time() - save_start_time
        print(f"Image saving time: {save_time:.2f} seconds")

    # Save transform parameters if requested
    if bspline_applied and output_transform_filepath and bspline_transform:
        try:
            # Save transform parameters
            sitk.WriteTransform(bspline_transform, output_transform_filepath)
            print(f"Transform saved to: {output_transform_filepath}")
        except Exception as e:
            print(f"Failed to save transform: {e}")

    # Calculate total workflow time
    total_time = time.time() - total_start_time
    print(f"\nTotal Registration Time: {total_time:.2f} seconds")

    # Collect registration information
    registration_info = {
        "hardware": hardware,
        "image_shape": reference_image.shape,
        "global_shift": global_shift,
        "global_shift_magnitude": np.linalg.norm(global_shift),
        "bspline_applied": bspline_applied,
        "bspline_grid_size": bspline_grid_size,
        "processing_time": {"loading": load_time, "total": total_time},
    }

    return fully_registered, registration_info


def main():
    # Example filepaths (replace with your actual paths)
    path = "/home/marcnol/grey/users/marcnol/test_HiM/testDataset"
    reference_path = path + os.sep + "scan_001_RT27_001_ROI_converted_decon_ch00.tif"
    target_path = path + os.sep + "scan_001_RT29_001_ROI_converted_decon_ch00.tif"
    output_path = (
        path
        + os.sep
        + "scan_001_RT29_001_ROI_converted_decon_ch00_bspline_registered.tif"
    )
    transform_path = path + os.sep + "scan_001_RT29_001_ROI_bspline_transform.tfm"

    # Perform advanced registration
    registered_image, reg_info = advanced_3d_registration(
        reference_path,
        target_path,
        output_path,
        global_upsample_factor=20,
        bspline_grid_size=(5, 5, 3),  # Control point density - adjust as needed
        bspline_iterations=200,  # More iterations for better convergence
        output_transform_filepath=transform_path,
    )

    # Print registration summary
    print("\nRegistration Summary:")
    print(f"Global shift: {reg_info['global_shift']}")
    print(f"Global shift magnitude: {reg_info['global_shift_magnitude']:.2f} pixels")
    print(
        f"B-spline registration {'applied' if reg_info['bspline_applied'] else 'failed'}"
    )
    print(f"Image registered and saved to {output_path}")
    if reg_info["bspline_applied"]:
        print(f"Transform saved to {transform_path}")


if __name__ == "__main__":
    main()
