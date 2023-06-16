import os
from pathlib import Path
import sys
from matplotlib import pyplot as plt, gridspec as gs, colors as mplcolors
from enum import Enum
import math

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.spectral_envi import *

from src.hsi_moss.moss2 import *

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")

stiff = STiff(
    basepath.joinpath("03-correction/t1s01A.corrected.darkcorrect.tif"),
    TiffOptions(),
    mtiffpath=basepath.joinpath("02-stiff/t1s01A.masks.tif"),
).read()


class Q(Enum):
    ORIGINAL_FLOAT = 0
    ORIGINAL_12BIT = 1


qs = [*list(range(5, 101, 5)), Q.ORIGINAL_12BIT.name, Q.ORIGINAL_FLOAT.name]
# qs = range(0, 101, 10)
files = []

for idx, q in enumerate(qs):
    name = basepath.joinpath(f'03a-decorrelation/t1s01A.jpg_{str(q).rjust(3,"0")}.tif')

    if not name.exists():
        st = stiff.copy()

        if q == Q.ORIGINAL_FLOAT.name:
            st.tiff_options = TiffOptions()
        else:
            if q == Q.ORIGINAL_12BIT.name:
                st.tiff_options = TiffOptions()
            else:
                st.tiff_options = TiffOptions(
                    compression_mode=("jpeg", q), rgb_only=False
                )
            st.cube = (
                RasterOps.normalize(stiff.cube, min=0, max=1) * 2**12 - 1
            ).astype(np.uint16)

        st.filepath = basepath.joinpath(
            f'03a-decorrelation/t1s01A.jpg_{str(q).rjust(3,"0")}.tif'
        )
        st.write_stiff()
    file_stats = os.stat(name.as_posix())
    files.append([name, file_stats.st_size / (1024 * 1024)])

    print(q)

n = len(qs)
cm = plt.cm.viridis
colors = cm(np.linspace(0, 1, n))

fig = plt.figure(constrained_layout=True, figsize=(11, 9))
spec = gs.GridSpec(
    ncols=2,
    nrows=3,
    figure=fig,
    width_ratios=[1, 1],
    height_ratios=[0.07, 0.7, 1],
    hspace=0,
    wspace=0,
)
GRID_ALPHA = 0.3
ax0 = fig.add_subplot(spec[1, 0])
ax0.set_yscale("log", base=2)
ax1 = fig.add_subplot(spec[2, :])
axzoom = fig.add_subplot(spec[1, 1])
axzoom.set_yscale("log", base=2)
axzoom.set_xlim(465, 503)
axzoom.set_ylim(0.0338, 0.039)
axzoom.set_xlabel("Wavelength (nm)")
axzoom.yaxis.tick_right()
cax = fig.add_subplot(spec[0, :])
cax.margins(x=0, y=0)

# axstats = fig.add_subplot(spec[2, 0])

for ax in fig.axes:
    ax.tick_params(direction="in")
    ax.grid(alpha=GRID_ALPHA)


ax0.set_xlabel("Wavelength (nm)")
ax0.set_ylabel("Reflectance Factor")
ax1

original_specimen = stiff.cube[stiff.masks["pot"] > 0]
middle_sample_idx = math.floor(len(original_specimen) / 2) + 1
original_signal = original_specimen[middle_sample_idx : middle_sample_idx + 20, :].mean(
    axis=0
)
rmspes = []
stats = []
for idx, q in enumerate(qs):
    st = STiff(
        basepath.joinpath(f'03a-decorrelation/t1s01A.jpg_{str(q).rjust(3,"0")}.tif'),
        mtiffpath=basepath.joinpath("02-stiff/t1s01A.masks.tif"),
    )

    specimen = st.cube[st.masks["pot"] > 0]
    if q != Q.ORIGINAL_FLOAT.name:
        specimen = specimen / (2**12 - 1)

    signal = specimen[middle_sample_idx : middle_sample_idx + 20, :].mean(axis=0)
    rmspe = 100 * np.sqrt(
        np.sum(np.square((signal / original_signal) - 1)) / len(signal)
    )
    rmspes.append(rmspe)

    diffs = original_specimen - specimen
    stats.append(np.std(diffs, axis=0))

    if q == Q.ORIGINAL_12BIT.name:
        ax0.plot(
            st.wavelengths,
            signal,
            label=f"Original 12-bit",
            color="red",
            lw=1.2,
        )
        axzoom.plot(
            st.wavelengths,
            signal,
            label=f"Original 12-bit",
            color="red",
            lw=1.2,
        )
    elif q == Q.ORIGINAL_FLOAT.name:
        ax0.plot(
            st.wavelengths,
            signal,
            label=f"Original Float",
            color="black",
            lw=1.2,
        )
        axzoom.plot(
            st.wavelengths,
            signal,
            label=f"Original Float",
            color="black",
            lw=1.2,
        )
    else:
        ax0.plot(st.wavelengths, signal, color=colors[idx], lw=0.4, alpha=1)
        axzoom.plot(st.wavelengths, signal, color=colors[idx], lw=0.4, alpha=1)

ax0.legend(prop={"size": 8}, loc="upper left", facecolor=(1, 1, 1, 1))
axzoom.legend(prop={"size": 8}, loc="upper left", facecolor=(1, 1, 1, 1))
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=5, vmax=100))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label(
    "Quality Level",
)
cb.ax.yaxis.labelpad = 8
cb.ax.xaxis.tick_top()
cb.ax.xaxis.set_label_position("top")

COLOR_SIZE = "#69b3a2"
COLOR_RMSE = "black"
# quality vs file size
ax1a = ax1.twinx()
ax1.set_xlabel("Quality Level")
ax1.set_ylabel("File Size (MB)", color=COLOR_SIZE)
ax1a.tick_params(axis="y", labelcolor=COLOR_SIZE)
files = np.array(files)
file_size = files[:, 1]
ax1.bar([str(q) for q in qs], file_size, color=COLOR_SIZE)
ax1.set_yscale("log")

# signal error plot
ax1a.plot([str(q) for q in qs], rmspes, color=COLOR_RMSE)
ax1a.ticklabel_format(axis="y", style="sci")
ax1a.set_ylabel("RMSPE (%)", color=COLOR_RMSE)
ax1a.tick_params(axis="y", labelcolor=COLOR_RMSE)
ax1a.set_yscale("log")


# signal statistics
# stats = np.array(stats)
# for idx, q in enumerate(qs):
#     axstats.bar(
#         stiff.wavelengths,
#         np.reshape(stats[idx], (-1)),
#         # s=4,
#         color=colors[idx],
#         align="edge",
#     )
# plt.show()
print("done")
fig.savefig(
    "scripts/jpg/jpg-stats.pdf",
    bbox_inches="tight",
    transparent=True,
)
fig.savefig(
    "scripts/jpg/jpg-stats.svg",
    bbox_inches="tight",
    transparent=True,
)
