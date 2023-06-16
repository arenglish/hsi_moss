from pathlib import Path
from .utils.logger import log
from .spectral_envi import *


def envi2cube(path: Path, darkref=None):
    if not path.exists():
        raise LookupError(f"{path.as_posix()} does not exist")

    with open(path.with_suffix(".hdr").as_posix(), "r") as f:
        hdr_text = f.read()

    im, wavelengths = read_envi(path.with_suffix(".hdr").as_posix())

    return (im, wavelengths, hdr_text)
