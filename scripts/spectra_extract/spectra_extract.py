import os
from pathlib import Path
import sys
import math
from matplotlib.widgets import Button
from matplotlib.figure import Figure
import numpy as np
import glob

sys.path.append(Path(os.getcwd()).as_posix())
from src.hsi_moss.raster import STiff
from src.hsi_moss.envi2tiff import envi2cube

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from enum import Enum


class IM_TYPE(Enum):
    STIFF = 0
    ENVI = 1


def extr(fname: str | Path, type: IM_TYPE):
    fig = plt.figure(layout="constrained", dpi=300, figsize=(4, 6))
    fig.suptitle("HSI View")
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(gs[0, 1])
    axclear = fig.add_subplot(gs[1, 0])
    axsave = fig.add_subplot(gs[1, 1])

    WHITECORRECT = False
    DARKCORRECT = True

    class Handler:
        path: Path
        fig: Figure
        stiff: STiff

        def plot(self, event):
            if event.inaxes == self.fig.axes[0]:
                x, y = (math.floor(event.xdata), math.floor(event.ydata))
                self.lastspectra = np.array(
                    [
                        self.stiff.wavelengths,
                        np.mean(
                            self.stiff.cube[y - 2 : y + 2, x - 2 : x + 2, :],
                            axis=(0, 1),
                        ),
                    ]
                ).T
                ax2.plot(self.lastspectra[:, 0], self.lastspectra[:, 1])
                plt.show()
                print("done")

        def save(self, event):
            path = self.stiff.filepath.with_suffix(".csv")
            newpath = path
            count = 0
            while Path(newpath.name).exists():
                newpath = path.with_stem(path.stem + "-" + str(count + 1))
                count = count + 1
            np.savetxt(
                newpath.name,
                self.lastspectra,
                delimiter=",",
                header="wavelengths,spectra",
            )

        def clear(self, event):
            self.fig.axes[1].clear()

    path = Path(fname)

    handler = Handler()
    handler.fig = fig

    bclear = Button(axclear, "Clear")
    bclear.on_clicked(handler.clear)
    bsave = Button(axsave, "Save Last")
    bsave.on_clicked(handler.save)

    if type is IM_TYPE.STIFF:
        handler.stiff = STiff(path)
    elif type is IM_TYPE.ENVI:
        (cube, wavelengths, metadata) = envi2cube(path)

        if DARKCORRECT is True:
            dpath = glob.glob(path.parent.joinpath("DARK*.raw").as_posix())
            dpath = dpath[0]
            (dcube, dwavelengths, dmetadata) = envi2cube(Path(dpath))
            dcube = np.mean(dcube, axis=0)
            cube = cube - dcube

        if WHITECORRECT is True:
            wpath = glob.glob(path.parent.joinpath("WHITE*.raw").as_posix())
            wpath = wpath[0]
            (wcube, wwavelengths, wmetadata) = envi2cube(Path(wpath))
            cube = (
                cube / np.mean(wcube, axis=0)
                if DARKCORRECT is False
                else (np.mean(wcube, axis=0) - dcube)
            )

        handler.stiff = STiff(path.with_suffix(".tif"))
        handler.stiff.filepath = path
        handler.stiff.cube = cube
        handler.stiff.wavelengths = wavelengths
        handler.stiff.metadata = metadata
        handler.stiff.render8bit()

    ax1.imshow(handler.stiff.rgb)
    cid = fig.canvas.mpl_connect("button_press_event", handler.plot)
    plt.show()


extr(
    Path(
        r"C:\Users\austi\OneDrive - NTNU\Coursework\UEF\ColorScienceLab\Session02\ColorChecker_Specim_V10E\capture\colorchecker_sample_0007.raw"
    ),
    IM_TYPE.ENVI,
)
