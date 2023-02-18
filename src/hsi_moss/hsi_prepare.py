from pathlib import Path, PurePath
from typing import SupportsInt, AnyStr
# importing PIL
from hsi_moss.spectral_envi import *
from hsi_moss.spectral_tiffs import write_stiff

def envi_to_stiff(hdrPath: PurePath, tiffPath: PurePath = None):
    if tiffPath is None:
        tiffPath = hdrPath.with_suffix('.tif')
        
    im, wavelengths = read_envi(hdrPath.as_posix())
    write_stiff(tiffPath.as_posix(), im, wavelengths, None)
