from CSL_homework_2 import *
import os
from pathlib import Path
import glob
import sys
from PIL import Image
import math
sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
import numpy as np
from matplotlib import pyplot as plt

stiff = STiff(r"C:\Users\austi\dev\hsi_moss\stiff_outputs\03-correction\t1s01A.corrected.tif").read()

# sensitivities
sensitivities = np.loadtxt('scripts/sensitivities.csv', delimiter=',')
wavelengths_s = sensitivities[:,0]
sensitivities = np.squeeze(RasterOps.normalize(sensitivities[:,1:]))

#illuminant
illuminants = CIE_light_sources()
D65 = RasterOps.normalize(illuminants[:,4])
wavelengths_ill = illuminants[:,0]

# colorchecker
cc24 = np.loadtxt('scripts/simulate_camera/SpectrumCC24.csv', skiprows=1, delimiter=',')
wavelengths_cc24 = cc24[:,0]
cc24 = cc24[:,1:]

# specimen SDs
specimens = []
for path in sorted(glob.glob(Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline\03b-specimen_mean").as_posix()+'/*.csv'), reverse=False):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    specimens.append(data[:,1])
wavelengths_specimens = data[:,0]
specimens = np.array(specimens)

# CMFs
cmfs = np.loadtxt('scripts/simulate_camera/lin2012xyz2e_1_7sf.csv', delimiter=',')
wavelengths_cmfs = cmfs[:,0]
cmfs = cmfs[:,1:]

# calculate XYZ of ColorChecker24
interp_10nm = {
    'wavelengths': wavelengths_cc24,
    'cc24': cc24,
    'specimens': np.array([np.interp(wavelengths_cc24, wavelengths_specimens, sp) for sp in specimens]).T,
    'illum': np.interp(wavelengths_cc24, wavelengths_ill, D65),
    'cmfs': np.array([np.interp(wavelengths_cc24, wavelengths_cmfs, sd) for sd in cmfs.T]),
    'sensitivities': np.array([np.interp(wavelengths_cc24, wavelengths_s, sd) for sd in sensitivities.T])
}
highpass = 5
interp_10nm['wavelengths'] = interp_10nm['wavelengths'][highpass:]
interp_10nm['cc24'] = interp_10nm['cc24'][highpass:]
interp_10nm['specimens'] = interp_10nm['specimens'][highpass:]
interp_10nm['illum'] = interp_10nm['illum'][highpass:]
interp_10nm['cmfs'] = interp_10nm['cmfs'][:,highpass:]
interp_10nm['sensitivities'] = interp_10nm['sensitivities'][:,highpass:]

samples_XYZ = np.squeeze(spim2XYZ(interp_10nm['specimens'], interp_10nm['wavelengths']))
samples_RGB = (np.squeeze(spim2rgb(interp_10nm['specimens'], interp_10nm['wavelengths']))*(2**8-1)).astype(np.uint8)
Image.fromarray(np.reshape(samples_RGB, (96,4,3)), mode='RGB').save('scripts/simulate_camera/outputs/specimen_rgb.png')

# calculate RGB rendering with camera
rgb = np.array([np.dot(sd*interp_10nm['illum'], interp_10nm['specimens']) for sd in interp_10nm['sensitivities']])
rgb_bias = np.vstack([np.ones(len(rgb.T)), rgb])
t = np.linalg.lstsq(rgb_bias.T, np.squeeze(samples_XYZ), rcond=-1)

rgb_norm = RasterOps.normalize(np.reshape([rgb.T], (96,4,3)), asinttype=np.uint8)
Image.fromarray(rgb_norm, mode='RGB').save('scripts/simulate_camera/outputs/specimen-cam_rgb.png')
XYZ_cam = rgb_bias.T @ t[0]

# visualize XYZ correction
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(XYZ_cam[:,0], XYZ_cam[:,1], XYZ_cam[:,2])
# ax.scatter(CC24_XYZ[:,0], CC24_XYZ[:,1], CC24_XYZ[:,2])
# ax.scatter(10*rgb.T[:,0], 10*rgb.T[:,1], 10*rgb.T[:,2])
# plt.show()


RGB_cam = (np.squeeze(XYZ2RGB(np.array([XYZ_cam])))*(2**8-1)).astype(np.uint8)
# visualize rgb correction
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(RGB_cam[:,0], RGB_cam[:,1], RGB_cam[:,2])
# ax.scatter(CC24_RGB[:,0], CC24_RGB[:,1], CC24_RGB[:,2])
# plt.show()

Image.fromarray(np.reshape(RGB_cam, (96,4,3)), mode='RGB').save('scripts/simulate_camera/outputs/specimen-cam_rgb_corrected.png')
np.savetxt('scripts/simulate_camera/cam_rgb2XYZ_specimen.csv', t[0], delimiter=',')

