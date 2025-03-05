#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-based local registration for Hi-M experiments.

This module implements a two-step registration process:
1. Global 3D rigid registration in XYZ
2. Local registration based on segmented masks

The approach is optimized for parallel processing and GPU acceleration when available.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

# Try importing GPU acceleration libraries
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy.ndimage import shift as cp_shift

    # Test GPU functionality
    a = cp.ones(1)
    GPU_AVAILABLE = True
    print("GPU acceleration enabled")
    del a
    cp.get_default_memory_pool().free_all_blocks()
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f"GPU acceleration not available: {e}")

"""# Try importing SimpleITK for additional registration methods
try:
    import SimpleITK as sitk

    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print(
        "SimpleITK not available. Install with 'pip install SimpleITK' for additional features."
    )
"""


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from file. Supports TIFF and NumPy (.npy) formats.

    Parameters:
    -----------
    filepath : str
        Path to the image file

    Returns:
    --------
    ndarray : The loaded image
    """
    try:
        # Check file extension to determine loading method
        if filepath.lower().endswith((".npy")):
            print(f"Loading NumPy file: {filepath}")
            image = np.load(filepath)
        else:
            # Assume TIFF or other format supported by skimage
            print(f"Loading TIFF image: {filepath}")
            image = io.imread(filepath)

        # Ensure 3D format
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 4:
            # Handle potential RGB/RGBA images
            image = image[:, :, :, 0] if image.shape[3] in [3, 4] else image.squeeze()

        print(f"Loaded image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        raise


def save_tiff_image(image: np.ndarray, filepath: str):
    """Save a 3D image as TIFF."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert to appropriate data type for TIFF
        # Float32 is commonly used for scientific images
        image_to_save = image.astype(np.float32)

        # Save the image using skimage's io.imsave
        # This maintains the 3D structure properly
        # The check_contrast parameter was added in newer versions of skimage
        try:
            io.imsave(filepath, image_to_save, check_contrast=False)
        except TypeError:
            # For older versions of skimage that don't have check_contrast
            io.imsave(filepath, image_to_save)

        print(f"Successfully saved image to {filepath}")
    except Exception as e:
        print(f"Error saving image {filepath}: {e}")
        raise


def global_3d_registration(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    upsample_factor: int = 10,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform global 3D rigid registration to align target to reference.

    Parameters:
    -----------
    reference_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image to register
    upsample_factor : int, optional
        Subpixel registration precision
    use_gpu : bool, optional
        Whether to use GPU acceleration if available

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
    ref_mean = reference_image.mean()
    ref_std = reference_image.std() + 1e-8
    target_mean = target_image.mean()
    target_std = target_image.std() + 1e-8

    ref_norm = (reference_image - ref_mean) / ref_std
    target_norm = (target_image - target_mean) / target_std

    # Use GPU if available and requested
    if GPU_AVAILABLE and use_gpu:
        try:
            # Transfer to GPU
            ref_norm_gpu = cp.asarray(ref_norm)
            target_norm_gpu = cp.asarray(target_norm)

            # Calculate shift using FFT-based cross-correlation on GPU
            # First compute FFTs
            ref_fft = cp.fft.fftn(ref_norm_gpu)
            target_fft = cp.fft.fftn(target_norm_gpu)

            # Compute cross-correlation
            cross_corr = cp.fft.ifftn(ref_fft * cp.conj(target_fft))

            # Find maximum correlation point
            max_idx = cp.unravel_index(cp.argmax(cp.abs(cross_corr)), cross_corr.shape)
            max_idx = cp.asnumpy(max_idx)

            # Convert to shift with respect to image center
            shift = np.array(
                [
                    (
                        (max_idx[0] - ref_norm.shape[0])
                        if max_idx[0] > ref_norm.shape[0] // 2
                        else max_idx[0]
                    ),
                    (
                        (max_idx[1] - ref_norm.shape[1])
                        if max_idx[1] > ref_norm.shape[1] // 2
                        else max_idx[1]
                    ),
                    (
                        (max_idx[2] - ref_norm.shape[2])
                        if max_idx[2] > ref_norm.shape[2] // 2
                        else max_idx[2]
                    ),
                ]
            )

            # Apply shift to get globally aligned image
            # Transfer back to GPU for shifting
            target_gpu = cp.asarray(target_image)
            registered_image_gpu = cp_shift(target_gpu, shift, mode="constant", cval=0)
            registered_image = cp.asnumpy(registered_image_gpu)

            # Free GPU memory
            del (
                ref_norm_gpu,
                target_norm_gpu,
                ref_fft,
                target_fft,
                cross_corr,
                target_gpu,
                registered_image_gpu,
            )
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU.")
            # Fall back to CPU
            shift, _, _ = phase_cross_correlation(
                ref_norm, target_norm, upsample_factor=upsample_factor
            )
            registered_image = ndimage.shift(
                target_image, shift, mode="constant", cval=0
            )
    else:
        # CPU-based registration
        shift, _, _ = phase_cross_correlation(
            ref_norm, target_norm, upsample_factor=upsample_factor
        )
        registered_image = ndimage.shift(target_image, shift, mode="constant", cval=0)

    print(f"Global registration complete in {time.time() - start_time:.2f} seconds")
    print(f"Global shift: {shift}")

    return registered_image, shift


def extract_mask_region(
    image: np.ndarray, mask: np.ndarray, label: int, padding: int = 5
) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    Extract a region from the image corresponding to a specific mask label.

    Parameters:
    -----------
    image : ndarray
        Full 3D image
    mask : ndarray
        Labeled mask image
    label : int
        Label to extract
    padding : int
        Additional padding around the region

    Returns:
    --------
    region : ndarray
        Extracted region from image
    slices : tuple
        Slice coordinates for the region (z_slice, y_slice, x_slice)
    """
    # Find the bounding box of the label
    position = np.where(mask == label)
    if len(position[0]) == 0:
        raise ValueError(f"Label {label} not found in mask")

    # Get min/max indices for each dimension
    z_min, z_max = max(0, position[0].min() - padding), min(
        image.shape[0], position[0].max() + padding + 1
    )
    y_min, y_max = max(0, position[1].min() - padding), min(
        image.shape[1], position[1].max() + padding + 1
    )
    x_min, x_max = max(0, position[2].min() - padding), min(
        image.shape[2], position[2].max() + padding + 1
    )

    # Create slices
    z_slice = slice(z_min, z_max)
    y_slice = slice(y_min, y_max)
    x_slice = slice(x_min, x_max)

    # Extract region
    region = image[z_slice, y_slice, x_slice]

    return region, (z_slice, y_slice, x_slice)


def register_mask_region(
    ref_region: np.ndarray, target_region: np.ndarray, upsample_factor: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register a small region extracted from the mask.

    Parameters:
    -----------
    ref_region : ndarray
        Reference region from the reference image
    target_region : ndarray
        Target region to register
    upsample_factor : int
        Upsampling factor for subpixel registration

    Returns:
    --------
    registered_region : ndarray
        Registered region
    shift : ndarray
        Local shift vector (z, y, x)
    """
    # Normalize regions for comparison
    ref_norm = (ref_region - ref_region.mean()) / (ref_region.std() + 1e-8)
    target_norm = (target_region - target_region.mean()) / (target_region.std() + 1e-8)

    # Calculate shift using phase correlation
    shift, _, _ = phase_cross_correlation(
        ref_norm, target_norm, upsample_factor=upsample_factor
    )

    # Apply shift
    registered_region = ndimage.shift(target_region, shift, mode="constant", cval=0)

    return registered_region, shift


def process_mask_region(
    ref_image: np.ndarray,
    target_image: np.ndarray,
    mask: np.ndarray,
    label: int,
    padding: int = 5,
    upsample_factor: int = 20,
) -> Tuple[Tuple[slice, slice, slice], np.ndarray, np.ndarray]:
    """
    Process a single mask region for registration.

    Parameters:
    -----------
    ref_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image (after global registration)
    mask : ndarray
        Labeled mask image
    label : int
        Label to process
    padding : int
        Additional padding around the region
    upsample_factor : int
        Upsampling factor for subpixel registration

    Returns:
    --------
    slices : tuple
        Slice coordinates for the region
    registered_region : ndarray
        Registered region
    shift : ndarray
        Local shift vector (z, y, x)
    """
    try:
        # Extract regions
        ref_region, slices = extract_mask_region(ref_image, mask, label, padding)
        target_region, _ = extract_mask_region(target_image, mask, label, padding)

        # Register regions
        registered_region, shift = register_mask_region(
            ref_region, target_region, upsample_factor
        )

        return slices, registered_region, shift
    except Exception as e:
        print(f"Error processing mask region {label}: {e}")
        return None


def parallel_mask_registration(
    ref_image: np.ndarray,
    target_image: np.ndarray,
    mask: np.ndarray,
    padding: int = 5,
    upsample_factor: int = 20,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Perform parallel registration of all mask regions.

    Parameters:
    -----------
    ref_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image (after global registration)
    mask : ndarray
        Labeled mask image
    padding : int
        Additional padding around each region
    upsample_factor : int
        Upsampling factor for subpixel registration
    max_workers : int, optional
        Maximum number of parallel workers

    Returns:
    --------
    registered_image : ndarray
        Locally registered image
    shift_map : dict
        Dictionary mapping label to local shift
    """
    # Get unique labels excluding background (0)
    labels = np.unique(mask)
    labels = labels[labels != 0]

    print(f"Processing {len(labels)} mask regions in parallel...")
    start_time = time.time()

    # Create output image (initialized with globally registered image)
    registered_image = target_image.copy()

    # Store shifts for each region
    shift_map = {}

    # Set number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(labels))

    # Process regions in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for label in labels:
            future = executor.submit(
                process_mask_region,
                ref_image,
                target_image,
                mask,
                label,
                padding,
                upsample_factor,
            )
            futures.append((label, future))

        # Process results as they complete
        for label, future in tqdm(futures, desc="Processing regions"):
            try:
                result = future.result()
                if result is not None:
                    slices, registered_region, shift = result
                    # Update the output image with the registered region
                    registered_image[slices] = registered_region
                    # Store the shift
                    shift_map[label] = shift
            except Exception as e:
                print(f"Error processing region {label}: {e}")

    print(f"Local registration complete in {time.time() - start_time:.2f} seconds")
    print(f"Successfully registered {len(shift_map)} of {len(labels)} regions")

    return registered_image, shift_map


def interpolate_shifts(
    mask: np.ndarray, shift_map: Dict[int, np.ndarray], method: str = "linear"
) -> np.ndarray:
    """
    Create a dense shift field by interpolating the sparse shifts.

    Parameters:
    -----------
    mask : ndarray
        Labeled mask image
    shift_map : dict
        Dictionary mapping label to local shift
    method : str
        Interpolation method ('linear', 'nearest', or 'cubic')

    Returns:
    --------
    shift_field : ndarray
        Dense shift field of shape (3, *mask.shape) for z, y, x components
    """
    from scipy.interpolate import griddata

    # Get mask shape
    z_size, y_size, x_size = mask.shape

    # Create coordinate grids
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(z_size), np.arange(y_size), np.arange(x_size), indexing="ij"
    )

    # Collect center points and shifts for each label
    centers = []
    shifts = []

    for label, shift in shift_map.items():
        # Find center of the labeled region
        points = np.where(mask == label)
        if len(points[0]) > 0:
            center_z = points[0].mean()
            center_y = points[1].mean()
            center_x = points[2].mean()
            centers.append([center_z, center_y, center_x])
            shifts.append(shift)

    # Convert to arrays
    centers = np.array(centers)
    shifts = np.array(shifts)

    if len(centers) < 4:
        print("Warning: Not enough points for interpolation. Using nearest neighbor.")
        method = "nearest"

    # Initialize shift field
    shift_field = np.zeros((3, z_size, y_size, x_size), dtype=np.float32)

    # Interpolate each component
    for i in range(3):  # z, y, x components
        shift_component = shifts[:, i]

        # Reshape grid for interpolation
        points_flat = z_grid.flatten(), y_grid.flatten(), x_grid.flatten()

        # Perform interpolation
        interpolated = griddata(
            centers,
            shift_component,
            np.vstack(points_flat).T,
            method=method,
            fill_value=0,
        )

        # Reshape to original dimensions
        shift_field[i] = interpolated.reshape(z_size, y_size, x_size)

    return shift_field


def apply_shift_field(
    image: np.ndarray, shift_field: np.ndarray, use_gpu: bool = True
) -> np.ndarray:
    """
    Apply a dense shift field to deform an image.

    Parameters:
    -----------
    image : ndarray
        Input image to deform
    shift_field : ndarray
        Dense shift field of shape (3, *image.shape)
    use_gpu : bool
        Whether to use GPU acceleration if available

    Returns:
    --------
    deformed_image : ndarray
        Deformed image
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate grids
    z_size, y_size, x_size = image.shape
    z_coords = np.arange(z_size)
    y_coords = np.arange(y_size)
    x_coords = np.arange(x_size)

    # Create sample points with shift applied
    z_grid, y_grid, x_grid = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    # Apply shifts
    z_sample = z_grid - shift_field[0]
    y_sample = y_grid - shift_field[1]
    x_sample = x_grid - shift_field[2]

    # Clip coordinates to image boundaries
    z_sample = np.clip(z_sample, 0, z_size - 1)
    y_sample = np.clip(y_sample, 0, y_size - 1)
    x_sample = np.clip(x_sample, 0, x_size - 1)

    # Use GPU if available and requested
    if GPU_AVAILABLE and use_gpu:
        try:
            # Transfer data to GPU
            image_gpu = cp.asarray(image)
            z_sample_gpu = cp.asarray(z_sample)
            y_sample_gpu = cp.asarray(y_sample)
            x_sample_gpu = cp.asarray(x_sample)

            # Use CuPy's map_coordinates for interpolation
            coords_gpu = cp.stack([z_sample_gpu, y_sample_gpu, x_sample_gpu], axis=0)
            deformed_image_gpu = cp_ndimage.map_coordinates(
                image_gpu,
                coords_gpu,
                order=1,  # Linear interpolation
                mode="constant",
                cval=0,
            )

            # Transfer back to CPU
            deformed_image = cp.asnumpy(deformed_image_gpu)

            # Free GPU memory
            del (
                image_gpu,
                z_sample_gpu,
                y_sample_gpu,
                x_sample_gpu,
                coords_gpu,
                deformed_image_gpu,
            )
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            print(f"GPU interpolation failed: {e}. Falling back to CPU.")
            # Fall back to CPU interpolation
            interpolator = RegularGridInterpolator(
                (z_coords, y_coords, x_coords), image, bounds_error=False, fill_value=0
            )

            # Reshape for interpolation
            points = np.stack(
                [z_sample.flatten(), y_sample.flatten(), x_sample.flatten()], axis=-1
            )
            deformed_image = interpolator(points).reshape(image.shape)
    else:
        # CPU interpolation
        interpolator = RegularGridInterpolator(
            (z_coords, y_coords, x_coords), image, bounds_error=False, fill_value=0
        )

        # Reshape for interpolation
        points = np.stack(
            [z_sample.flatten(), y_sample.flatten(), x_sample.flatten()], axis=-1
        )
        deformed_image = interpolator(points).reshape(image.shape)

    return deformed_image


def mask_based_registration(
    reference_filepath: str,
    target_filepath: str,
    mask_filepath: str,
    output_filepath: Optional[str] = None,
    global_upsample_factor: int = 20,
    local_upsample_factor: int = 100,
    padding: int = 10,
    max_workers: Optional[int] = None,
    interpolation_method: str = "linear",
    use_gpu: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform registration using a mask-based approach.

    Parameters:
    -----------
    reference_filepath : str
        Path to reference TIFF image
    target_filepath : str
        Path to target TIFF image to register
    mask_filepath : str
        Path to labeled mask TIFF image
    output_filepath : str, optional
        Path to save registered image
    global_upsample_factor : int
        Upsampling factor for global registration
    local_upsample_factor : int
        Upsampling factor for local registration
    padding : int
        Additional padding around mask regions
    max_workers : int, optional
        Maximum number of parallel workers
    interpolation_method : str
        Method for interpolating shifts ('linear', 'nearest', or 'cubic')
    use_gpu : bool
        Whether to use GPU acceleration if available

    Returns:
    --------
    registered_image : ndarray
        Fully registered image
    registration_info : dict
        Dictionary containing registration parameters and results
    """
    # Start overall timing
    total_start_time = time.time()

    # Load images
    print("Loading images...")
    load_start_time = time.time()
    reference_image = load_image(reference_filepath)
    target_image = load_image(target_filepath)
    mask_image = load_image(mask_filepath).astype(np.int32)
    load_time = time.time() - load_start_time
    print(f"Image loading time: {load_time:.2f} seconds")
    print(
        f"Image shapes: Reference {reference_image.shape}, Target {target_image.shape}, Mask {mask_image.shape}"
    )

    # Step 1: Global 3D registration
    print("\n--- STEP 1: GLOBAL REGISTRATION ---")
    globally_registered, global_shift = global_3d_registration(
        reference_image,
        target_image,
        upsample_factor=global_upsample_factor,
        use_gpu=use_gpu,
    )

    # Step 2: Mask-based local registration
    print("\n--- STEP 2: MASK-BASED LOCAL REGISTRATION ---")
    if mask_image.max() == 0:
        print("Warning: Mask is empty. Skipping local registration.")
        fully_registered = globally_registered
        shift_map = {}
    else:
        # Process mask regions in parallel
        locally_registered, shift_map = parallel_mask_registration(
            reference_image,
            globally_registered,
            mask_image,
            padding=padding,
            upsample_factor=local_upsample_factor,
            max_workers=max_workers,
        )

        # Step 3: Create dense deformation field by interpolating local shifts
        print("\n--- STEP 3: CREATING DENSE DEFORMATION FIELD ---")
        shift_field = interpolate_shifts(
            mask_image, shift_map, method=interpolation_method
        )

        # Step 4: Apply dense deformation field to the globally registered image
        print("\n--- STEP 4: APPLYING DEFORMATION FIELD ---")
        fully_registered = apply_shift_field(
            globally_registered, shift_field, use_gpu=use_gpu
        )

    # Save registered image
    if output_filepath:
        print("\nSaving registered image...")
        save_start_time = time.time()

        try:
            # Ensure the image has proper dimensions and data type for TIFF
            output_image = fully_registered.astype(np.float32)

            # Make sure the file extension is .tif or .tiff
            if not (
                output_filepath.lower().endswith(".tif")
                or output_filepath.lower().endswith(".tiff")
            ):
                output_filepath = output_filepath + ".tif"
                print(f"Added .tif extension to output path: {output_filepath}")

            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True
            )

            # Save the image
            save_tiff_image(output_image, output_filepath)

            # Verify the file was created
            if os.path.exists(output_filepath):
                file_size = os.path.getsize(output_filepath) / (
                    1024 * 1024
                )  # Convert to MB
                print(f"Verified: File saved successfully ({file_size:.2f} MB)")
            else:
                print("Warning: Output file not found after save operation")

        except Exception as e:
            print(f"Error saving registered image: {e}")
            import traceback

            traceback.print_exc()

        save_time = time.time() - save_start_time
        print(f"Image saving time: {save_time:.2f} seconds")

    # Calculate total workflow time
    total_time = time.time() - total_start_time
    print(f"\nTotal Registration Time: {total_time:.2f} seconds")

    # Collect registration information
    registration_info = {
        "hardware": "GPU" if GPU_AVAILABLE and use_gpu else "CPU",
        "image_shape": reference_image.shape,
        "global_shift": global_shift.tolist(),
        "global_shift_magnitude": float(np.linalg.norm(global_shift)),
        "local_regions_processed": len(shift_map),
        "processing_time": {"loading": load_time, "total": total_time},
    }

    return fully_registered, registration_info


def main():
    # Example filepaths (replace with your actual paths)
    path = "/home/marcnol/grey/users/marcnol/test_HiM/testDataset_deformed_brains"
    reference_path = os.path.join(path, "scan_005_RT2_043_ROI_ch00.tif")
    target_path = os.path.join(path, "scan_001_RT33_043_ROI_ch00.tif")
    mask_path = os.path.join(
        path, "mask_3d/data/scan_001_mask0_043_ROI_ch01_3Dmasks.npy"
    )
    output_path = os.path.join(path, "registered_output.tif")

    # Determine optimal number of workers based on system
    max_workers = min(
        multiprocessing.cpu_count(), 16
    )  # Limit to 16 to avoid memory issues

    # Perform mask-based registration
    registered_image, reg_info = mask_based_registration(
        reference_path,
        target_path,
        mask_path,
        output_path,
        global_upsample_factor=20,
        local_upsample_factor=100,
        padding=10,
        max_workers=max_workers,
        interpolation_method="linear",
        use_gpu=True,
    )

    # Print registration summary
    print("\nRegistration Summary:")
    print(f"Global shift: {reg_info['global_shift']}")
    print(f"Global shift magnitude: {reg_info['global_shift_magnitude']:.2f} pixels")
    print(f"Processed {reg_info['local_regions_processed']} local regions")
    print(f"Total processing time: {reg_info['processing_time']['total']:.2f} seconds")
    print(f"Image registered and saved to {output_path}")


if __name__ == "__main__":
    main()
