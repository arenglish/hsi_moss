import importlib_resources
import numpy as np

my_resources = importlib_resources.files("hsi_moss") / "camera_simulation"

camRGB2XYZ = np.loadtxt((my_resources / 'cam_rgb2XYZ.csv').as_posix(), delimiter=',')
camRGB2XYZ_specimen = np.loadtxt((my_resources / 'cam_rgb2XYZ_specimen.csv').as_posix(), delimiter=',')
sensitivities = np.loadtxt((my_resources / 'sensitivities.csv').as_posix(), delimiter=',')