from pathlib import Path
from .spectral_tiffs import *
import numpy as np
from typing import Tuple
from typing import List
from progress.bar import Bar
from utils.logger import log
from utils.logger import log


def get_rgb(spim, wavelengths, channels: Tuple):
    log.info(f"Selecting bands closest to {channels}")
    log.info(f"Source bands: {wavelengths}")
    bands = (
        (np.abs(wavelengths - channels[0])).argmin(),
        (np.abs(wavelengths - channels[1])).argmin(),
        (np.abs(wavelengths - channels[2])).argmin(),
    )
    log.info(f"Selected: {bands}")
    rgb = np.zeros((spim.shape[0], spim.shape[1], 3))
    rgb[:, :, 0] = spim[:, :, bands[0]]
    rgb[:, :, 1] = spim[:, :, bands[1]]
    rgb[:, :, 2] = spim[:, :, bands[2]]

    # apply gamma
    rgb = rgb ** (1 / 2.2)

    min = np.amin(rgb)
    max = np.amax(rgb)
    rgb = ((2**8 - 1) * (rgb - min) / (max - min)).astype(np.uint8)

    return rgb


def generate_tiff_previews(paths: List[Path], channels: Tuple = (650, 550, 450)):
    bar = Bar("Generating RGB previews for tiffs", max=len(paths))

    for path in paths:
        bar.next()
        spim, wavelengths, rgb, metadata = read_stiff(path.as_posix())
        if rgb is not None:
            log.warning(f"{path.as_posix()} already has an RGB preview")
            continue
        rgb = get_rgb(spim, wavelengths, channels)
        write_stiff(path.as_posix(), spim, wavelengths, rgb, compression=8)
    bar.finish()

    return paths


def cube2tiff(hsi, rgb, wavelengths, tiffpathstr: str, overwrite=False, compression=8):
    tiffpath = Path(tiffpathstr)

    if tiffpath.exists():
        log.warning(f"{tiffpath.as_posix()} already exists")
        if overwrite is False:
            return tiffpath

    write_stiff(tiffpath.as_posix(), hsi, wavelengths, rgb=rgb, compression=compression)
    return tiffpath
