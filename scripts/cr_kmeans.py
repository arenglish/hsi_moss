import os
from pathlib import Path
import sys
sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
import numpy as np
from matplotlib import pyplot as plt


stiff = STiff(r"C:\Users\austi\dev\hsi_moss\stiff_outputs\04-continuum_removal\t1s01A.cr1.tif").read()

segmented, centers = RasterOps.segment_kmeans(stiff.cube, 4, 1000)
plt.imshow(segmented)