import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian
from skimage.measure import regionprops, label
import tifffile as tiff


def get_centroid(mask):
    region = regionprops(label(mask))[0]
    return region.centroid


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
        raise ValueError("Mask should contain only one connected component.")

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
