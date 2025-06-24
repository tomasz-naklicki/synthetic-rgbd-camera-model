import numpy as np
from scipy.ndimage import minimum_filter


def circular_kernel(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def filter_depth_with_local_min_scipy(
    depth_img: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    mask_valid = np.isfinite(depth_img) & (depth_img > 0)
    depth_inf = np.where(mask_valid, depth_img, np.inf)

    local_min = minimum_filter(
        depth_inf, size=kernel_size, mode="constant", cval=np.inf
    )

    filtered = np.where(mask_valid, np.minimum(depth_img, local_min), depth_img)
    return filtered.astype(np.float32)
