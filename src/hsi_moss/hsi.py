import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def get_nearest_band_idx(wavelengths, target):
    return (np.abs(wavelengths - target)).argmin()


def normalize(raster, intrinsic_scale=False):
    dtype = raster.dtype
    max = np.amax(raster)
    min = np.amin(raster)

    max_allowed = np.iinfo(raster.dtype).max

    return np.multiply(max_allowed, (raster - min) / (max - min)).astype(dtype)


def scale_histogram(raster, bins, range, pct):
    hist = np.histogram(raster, bins=bins, range=range)
    total = raster.size
    count = 0
    i = 0

    while count < total * pct:
        count += hist[0][i]
        i += 1

    max_intensity = hist[1][i]
    raster[raster > max_intensity] = 0
    return normalize(raster)


def remove_specular(raster, threshold=None, threshold_pct=0.1):
    max = np.iinfo(raster.dtype).max

    if threshold is None:
        thresh = max - max * threshold_pct
    else:
        thresh = threshold

    max_allowed = max - thresh

    raster[raster > max_allowed] = max_allowed
    return raster


def average_and_stretch(data: ArrayLike, stretch_to: Tuple[int]):
    if type(data) is not np.ndarray:
        data = np.array(data)

    data = np.mean(data, axis=tuple(range(0, len(data.shape) - 1)))
    return np.broadcast_to(data, stretch_to)


def hsi_correction(hsi: np.ndarray, whiteref: np.ndarray, darkref: ArrayLike = None):
    """
    Perform dark and white corrections on hyperspectral cube.

    Parameters
    ----------
    hsi : array_like (O,M,N)
                    Hyperspectral cube to correct.
    whiteref : array_like (...,...,N)
                    Image of white reflectance sample.
    darkref : array_like (...,...,N), optional
                    Camera's dark image.  If None (default), no dark correction is
                    performed.

    Returns
    -------
    cube : ndarray, (O,M,N)
    """

    if whiteref.shape != hsi.shape:
        whiteref = average_and_stretch(whiteref, hsi.shape)

    if darkref is not None and darkref.shape != hsi.shape:
        darkref = average_and_stretch(darkref, hsi.shape)

    return hsi / whiteref if darkref is None else (hsi - darkref) / (whiteref - darkref)
