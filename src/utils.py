import numpy as np
from scipy.ndimage import minimum_filter


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def filter_depth_with_local_min_scipy(
    depth_img: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Apply a local minimum filter to a depth image, preserving existing finite values.

    For each pixel, if it is valid (finite and >0), replaces its value
    with the minimum depth within a local neighborhood; invalid pixels
    remain unchanged.

    Args:
        depth_img (np.ndarray): 2D array of depth values (float32 or float64).
        kernel_size (int, optional): Size of a kernel used
            for minimum filtering. Defaults to 3.

    Returns:
        np.ndarray: Filtered depth image as float32, same shape as input.
    """
    mask_valid = np.isfinite(depth_img) & (depth_img > 0)
    depth_inf = np.where(mask_valid, depth_img, np.inf)

    local_min = minimum_filter(
        depth_inf, size=kernel_size, mode="constant", cval=np.inf
    )

    filtered = np.where(mask_valid, np.minimum(depth_img, local_min), depth_img)
    return filtered.astype(np.float32)
