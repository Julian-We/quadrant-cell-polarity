import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian
from skimage.measure import regionprops, label
import tifffile as tiff
import hashlib
import pandas as pd
from pathlib import Path


def get_centroid(mask):
    region = regionprops(label(mask))[0]
    return region.centroid


def select_center_object(segmentation):
    if isinstance(segmentation, np.ndarray):
        labeled = label(segmentation)
        regions = regionprops(labeled)
        if len(regions) == 0:
            return segmentation
        center = np.array(segmentation.shape) // 2
        distances = [np.linalg.norm(np.array(r.centroid) - center) for r in regions]
        closest_region = regions[np.argmin(distances)]
        selected_mask = (labeled == closest_region.label).astype(int)
        return selected_mask


def get_polarity(intensity_image, mask, sigma=5):
    """Get the polarity normalized vector and magnitude by controid/weighted_centroid on a gaussian blurred image
    param intensity_image: 2D numpy array of the intensity image
    param mask: 2D numpy array of the binary mask
    param sigma: float, the standard deviation for Gaussian kernel

    I suggest using a high sigma value to smooth the image and avoid noise, but not too high to lose the polarity information. A value of 5 is a good starting point.
    """

    blurred_image = gaussian(intensity_image, sigma=sigma)

    label_mask = label(mask)

    if np.max(label_mask) == 1:
        pass
    elif np.max(label_mask) == 0:
        return (None, None), None
    else:
        print(
            "WARNING: Mask should contain only one connected component. Selecting the most center one"
        )
        label_mask = select_center_object(label_mask)

    controid = regionprops(label_mask)[0].centroid
    weighted_centroid = regionprops(label_mask, intensity_image=blurred_image)[
        0
    ].weighted_centroid

    vector = np.array(weighted_centroid) - np.array(controid)

    norm = np.linalg.norm(vector)

    return (vector / norm, norm) if norm != 0 else ((None, None), None)


def get_angled_unitvector(v, angle):
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def generate_quadrant_mask(shape, centroid, v1, v2):
    """Fill the wedge spanned by v1 and v2, anchored at centroid."""
    rows, cols = np.indices(shape)
    d = np.stack([rows - centroid[0], cols - centroid[1]], axis=-1)
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)

    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    if cross(v1, v2) < 0:  # ensure v1 -> v2 is the positive sweep
        v1, v2 = v2, v1

    return ((cross(v1, d) >= 0) & (cross(d, v2) >= 0)).astype(float)


def get_angle_between_vectors(v1, v2):
    return np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))


def get_ring_mask(cell_mask, thickness=2, pixel_size=1):
    """Calculates membrane mask from cell mask as a 'Ring'"""
    inside_dist = ndi.distance_transform_edt(cell_mask > 0) * pixel_size
    inside_area = inside_dist <= thickness

    return inside_area.astype(bool) & (cell_mask > 0)


def read_image(path):
    """Reads a tiff image from the given path and returns it as a numpy array"""
    return tiff.imread(path)


def lsm_pixel_size(path):
    """Return pixel size in microns, asserting square pixels."""
    with tiff.TiffFile(path) as tif:
        m = tif.lsm_metadata
        x, y = m["VoxelSizeX"] * 1e6, m["VoxelSizeY"] * 1e6
    assert abs(x - y) < 1e-9, f"non-square pixels: {x} != {y}"
    return x


def time_clipper(image, index_start, index_end):
    """Clip a 3D image along the time axis (axis=0)"""
    if image.ndim != 3:
        raise ValueError("Image must be 3D (time, height, width)")
    return image[index_start:index_end]


def get_unique_id(image):
    sha_hash = hashlib.sha256()
    if isinstance(image, str):
        with open(image, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha_hash.update(chunk)
    elif isinstance(image, np.ndarray):
        sha_hash.update(image.tobytes())
    else:
        raise ValueError("Input must be a file path or a numpy array.")
    return sha_hash.hexdigest()[:8]


def export_data(list_of_dicts, output_path: str | Path | None = None):
    df = pd.DataFrame(list_of_dicts)
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df


def apply_gaussian_filter(image, sigma=1):
    """Apply Gaussian blur to a 2D image."""
    return gaussian(image, sigma=sigma)


def mask_checker(list_of_masks):
    """Checks is all masks/segmentation are roughly the same size and shape. Returns True if they are, False otherwise."""
    mask_areas = [np.sum(mask.astype(bool)) for mask in list_of_masks]
    median_area = np.median(mask_areas)
    max_deviation = 1.5 * median_area  # Allow 50% deviation
    biggest_mask = np.argmax(mask_areas)
    if biggest_mask is not None and mask_areas[biggest_mask] > max_deviation:
        print(
            f"WARNING: Mask {biggest_mask} is significantly larger than the median area ({mask_areas[biggest_mask]} vs {median_area}; {mask_areas[biggest_mask] / mask_areas}x)."
        )
        return False
    else:
        return True
