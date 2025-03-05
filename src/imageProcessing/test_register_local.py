#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple

# import dask.array as da
import dask.distributed
import numpy as np
from scipy import ndimage
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation


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


def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Mutual Information between two images

    Parameters:
    -----------
    img1, img2 : ndarray
        Input images of same shape
    bins : int, optional
        Number of bins for histogram computation

    Returns:
    --------
    float: Mutual Information score
    """
    # Joint histogram
    hist_2d, _ = np.histogramdd([img1.ravel(), img2.ravel()], bins=[bins, bins])

    # Marginal histograms
    hist_1 = np.sum(hist_2d, axis=1)
    hist_2 = np.sum(hist_2d, axis=0)

    # Probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = hist_1 / float(np.sum(hist_1))
    py = hist_2 / float(np.sum(hist_2))

    # Compute MI
    px_py = px[:, np.newaxis] * py[np.newaxis, :]
    nzx = pxy > 0
    return np.sum(pxy[nzx] * np.log(pxy[nzx] / px_py[nzx]))


def block_registration_3d(
    reference_block: np.ndarray,
    target_block: np.ndarray,
    metric: str = "mi",
    upsample_factor: int = 10,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Perform block-wise 3D image registration

    Parameters:
    -----------
    reference_block : ndarray
        Reference 3D image block
    target_block : ndarray
        Target 3D image block to register
    metric : str, optional
        Registration metric ('mi', 'ncc', 'ssim')
    upsample_factor : int, optional
        Subpixel registration precision

    Returns:
    --------
    shift : ndarray
        Estimated shift in ZXY
    quality : float
        Registration quality metric
    shifted_block : ndarray
        Registered block
    """
    # Normalize blocks for comparison
    ref_norm = (reference_block - reference_block.mean()) / (
        reference_block.std() + 1e-8
    )
    target_norm = (target_block - target_block.mean()) / (target_block.std() + 1e-8)

    # Estimate shifts using phase correlation
    shifts, _, _ = phase_cross_correlation(
        ref_norm, target_norm, upsample_factor=upsample_factor
    )

    # Apply calculated shifts
    shifted_block = ndimage.shift(target_block, shifts, mode="nearest")

    # Compute registration quality metric
    if metric == "mi":
        quality = mutual_information(reference_block, shifted_block)
    elif metric == "ssim":
        quality = ssim(
            reference_block,
            shifted_block,
            data_range=shifted_block.max() - shifted_block.min(),
        )
    else:
        quality = np.corrcoef(reference_block.ravel(), shifted_block.ravel())[0, 1]

    return shifts, quality, shifted_block


def parallel_block_registration_3d(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    block_size: Tuple[int, int, int] = (64, 64, 16),
    overlap: int = 8,
    metric: str = "mi",
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Tuple[int, int, int], float]]]:
    """
    Perform parallel block-wise 3D image registration

    Parameters:
    -----------
    reference_image : ndarray
        Reference 3D image
    target_image : ndarray
        Target 3D image to register
    block_size : tuple, optional
        Size of registration blocks
    overlap : int, optional
        Overlap between blocks for smooth registration
    metric : str, optional
        Registration metric

    Returns:
    --------
    registered_image : ndarray
        Fully registered image
    shift_matrix : ndarray
        Matrix of shifts for each block
    block_metrics : list
        List of (shift, quality) for each block
    """
    # Validate input images
    assert reference_image.shape == target_image.shape, "Images must have same shape"

    # Initialize output image and tracking matrices
    registered_image = np.zeros_like(target_image)
    shift_matrix = np.zeros(
        (
            reference_image.shape[0],
            reference_image.shape[1],
            reference_image.shape[2],
            3,
        )
    )
    block_metrics = []

    # Use Dask for parallel processing
    with dask.distributed.Client() as client:
        futures = []

        # Iterate through blocks with overlap
        for z in range(0, reference_image.shape[0], block_size[0] - overlap):
            for y in range(0, reference_image.shape[1], block_size[1] - overlap):
                for x in range(0, reference_image.shape[2], block_size[2] - overlap):
                    # Extract blocks
                    ref_block = reference_image[
                        z : z + block_size[0],
                        y : y + block_size[1],
                        x : x + block_size[2],
                    ]
                    target_block = target_image[
                        z : z + block_size[0],
                        y : y + block_size[1],
                        x : x + block_size[2],
                    ]

                    # Submit registration task
                    future = client.submit(
                        block_registration_3d, ref_block, target_block, metric=metric
                    )
                    futures.append((future, (z, y, x)))

        # Collect and process results
        for future, (z, y, x) in futures:
            shifts, quality, shifted_block = future.result()

            # Update registered image
            block_slice = (
                slice(z, z + block_size[0]),
                slice(y, y + block_size[1]),
                slice(x, x + block_size[2]),
            )
            registered_image[block_slice] = shifted_block

            # Store shift and quality
            shift_matrix[block_slice[0], block_slice[1], block_slice[2]] = shifts
            block_metrics.append(((z, y, x), shifts, quality))

    return registered_image, shift_matrix, block_metrics


def register_images(
    reference_filepath: str,
    target_filepath: str,
    output_filepath: Optional[str] = None,
    block_size: Tuple[int, int, int] = (64, 64, 16),
    metric: str = "mi",
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Tuple[int, int, int], float]]]:
    """
    Comprehensive image registration workflow

    Parameters:
    -----------
    reference_filepath : str
        Path to reference TIFF image
    target_filepath : str
        Path to target TIFF image to register
    output_filepath : str, optional
        Path to save registered image
    block_size : tuple, optional
        Registration block size
    metric : str, optional
        Registration metric

    Returns:
    --------
    registered_image : ndarray
        Fully registered image
    shift_matrix : ndarray
        Matrix of shifts for each block
    block_metrics : list
        List of (block_position, shifts, quality) for each block
    """
    # Load images
    reference_image = load_tiff_image(reference_filepath)
    target_image = load_tiff_image(target_filepath)

    # Perform registration
    registered_image, shift_matrix, block_metrics = parallel_block_registration_3d(
        reference_image, target_image, block_size=block_size, metric=metric
    )

    # Save registered image if output path provided
    if output_filepath:
        save_tiff_image(registered_image, output_filepath)

    return registered_image, shift_matrix, block_metrics


# Example usage
def main():
    # Example filepaths (replace with your actual paths)
    path = "/home/marcnol/grey/users/marcnol/test_HiM/testDataset"
    reference_path = path + os.sep + "scan_001_RT27_001_ROI_converted_decon_ch00.tif"
    target_path = path + os.sep + "scan_001_RT29_001_ROI_converted_decon_ch00.tif"
    output_path = (
        path + os.sep + "scan_001_RT29_001_ROI_converted_decon_ch00_registered.tif"
    )

    # Perform registration
    registered_image, shift_matrix, block_metrics = register_images(
        reference_path, target_path, output_path, block_size=(64, 64, 16), metric="mi"
    )

    # Print registration summary
    print(f"Image registered and saved to {output_path}")
    print(f"Total blocks processed: {len(block_metrics)}")
    print("Average block shifts:", np.mean([m[1] for m in block_metrics], axis=0))
    print("Average block quality:", np.mean([m[2] for m in block_metrics]))


if __name__ == "__main__":
    main()
