from glob import glob
from src.hsi_moss.raster import *

for f in glob(r"I:\moss_data\Austin moss 2023\Moss\pipeline\**\*.tif"):
    if "masks" not in f:
        stiff = STiff(f, TiffOptions(8, False)).read()
        stiff.write_stiff()
