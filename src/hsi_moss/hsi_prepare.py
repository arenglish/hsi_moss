from pathlib import Path, PurePath
from typing import SupportsInt, AnyStr
# importing PIL
from spectral_envi import *
from spectral_tiffs import write_stiff
 

def open_image(path: PurePath):
    return open(path.absolute())

def envi_to_stiff(hdrPath: PurePath, tiffPath: PurePath = None):
    if tiffName is None:
        tiffName = hdrPath.with_suffix('.tif')
        
    im, wavelengths = read_envi(hdrPath.absolute())
    write_stiff(tiffPath.name, im, wavelengths)
