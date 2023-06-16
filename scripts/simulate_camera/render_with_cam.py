import os
from CSL_homework_2 import *
from pathlib import Path
import sys
sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
import numpy as np
from matplotlib import pyplot as plt


stiff = STiff(r"C:\Users\austi\dev\hsi_moss\stiff_outputs\03-correction\t1s01A.corrected.tif").read()
sensitivities = np.loadtxt('scripts/sensitivities.csv', delimiter=',')
illuminants = CIE_light_sources()
D65 = RasterOps.normalize(illuminants[:,4])*.6
wavelengths_ill = illuminants[:,0]
wavelengths_s = sensitivities[:,0]
sensitivities = RasterOps.normalize(sensitivities[:,1:])


# interpolate illuminant down to match sensitivity samples
illuminant = np.interp(wavelengths_s, wavelengths_ill, D65)

raster = RasterOps.linear_interpolate(stiff.cube, stiff.wavelengths, wavelengths_s)
render = RasterOps.render_with_sensitivities(reflectance=raster, illuminant_spd=illuminant, sensitivities=sensitivities)
camRGB2XYZ = np.loadtxt('scripts/simulate_camera/cam_rgb2XYZ_specimen.csv', delimiter=',')
render_shape = render.shape
render = np.reshape(render, (render.shape[0]*render.shape[1],3))
render_bias = np.vstack([np.ones(len(render)), render.T])
renderXYZ = np.reshape(render_bias.T @ camRGB2XYZ, render_shape)

renderRGB = np.squeeze(XYZ2RGB(renderXYZ))*(2**8-1)
# renderRGB[renderRGB>(2**8-1)] = (2**8-1)
renderRGB = renderRGB.astype(np.uint8)
plt.imshow(renderRGB)
plt.show()
print('done')