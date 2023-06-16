from src.hsi_moss.raster import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import math

stiffpath = (
    r"I:\moss_data\Austin moss 2023\Moss\pipeline\stiff_original\t2swhite387.tif"
)


def mtiff_gen():
    path = Path(stiffpath)
    mtiff_path = path.with_stem(path.stem + ".masks")
    stiff = STiff(stiffpath, TiffOptions(0, True)).read(
        read_masks=mtiff_path.as_posix()
    )

    plt.figure()
    radius = float(input("Mask Radius: "))
    name = input("Mask Name: ")
    print("Pick a center point")
    plt.imshow(stiff.rgb)
    point = np.asarray(plt.ginput(1, timeout=-1))[0]
    plt.close()

    mask = np.zeros_like(stiff.rgb[:, :, 0])
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            loc = (
                math.sqrt((x - int(point[1])) ** 2 + (y - int(point[0])) ** 2)
                - radius**2
            )
            if loc < 0:
                mask[x, y] = 1
    stiff.masks[name] = mask
    preview = stiff.rgb.copy()
    preview[mask == 0] = 0
    plt.figure()
    plt.imshow(preview)
    confirm = input("Confirm with any key: ")
    plt.close()
    stiff.write_mtiff(mtiff_path)


mtiff_gen()
