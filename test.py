from src.hsi_moss.raster import *
from pathlib import Path

basepath = Path(r"C:\Users\austi\dev\hsi_moss\stiff_outputs")
stiff = STiff(basepath.joinpath(r"02-stiff\t1s01A.tif"), TiffOptions())
