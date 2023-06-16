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
    PCA = 1


# qs = range(0, 101, 10)
files = []

n_components = 30
raster, components, mean, pca = RasterOps.cube2pca(
    stiff.cube, n_components=n_components, n_components_keep=n_components
)

files = []

for idx, numc in enumerate(range(1, len(components) + 1)):
    name = basepath.joinpath(
        f"03a-decorrelation/t1s01A.pca_{str(numc).rjust(2, '0')}.tif"
    )
    if not name.exists():
        st = stiff.copy()
        st.cube = raster.astype(np.float32)[:, :, :numc]
        st.type = STIFF_TYPE.PCA
        st.extras = {
            "components": components[:numc, :],
            "mean": np.array([mean]),
            "wavelengths": stiff.wavelengths,
        }

        st.filepath = basepath.joinpath(
            f"03a-decorrelation/t1s01A.pca_{str(numc).rjust(2, '0')}.tif"
        )
        st.wavelengths = np.array(list(range(numc)), dtype=np.float32)
        st.render8bit()
        st.write_stiff()

    file_stats = os.stat(name.as_posix())
    files.append([Q.PCA.name, name, file_stats.st_size / (1024 * 1024)])
    print(f"Component {idx}")

files.append(
    [
        Q.ORIGINAL_FLOAT.name,
        stiff.filepath,
        os.stat(stiff.filepath.as_posix()).st_size / (1024 * 1024),
    ]
)
cm = plt.cm.gist_ncar
colors = cm(np.linspace(0, 1, n_components))
cm = mplcolors.ListedColormap(colors)

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
ax0.set_ylim(2**-6, 1)

ax1 = fig.add_subplot(spec[2, :])
axzoom = fig.add_subplot(spec[1, 1])
axzoom.set_yscale("log", base=2)
axzoom.set_xlim(625, 690)
axzoom.set_ylim(0.05, 0.1)
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
original_signal = original_specimen[middle_sample_idx, :]
rmspes = []
stats = []
for idx, (type, name, size) in enumerate(files[:-1]):
    st = STiff(
        name,
        mtiffpath=basepath.joinpath("02-stiff/t1s01A.masks.tif"),
    )
    st.cube = st.reconstruct_from_pca()

    specimen = st.cube[st.masks["pot"] > 0]
    signal = specimen[middle_sample_idx, :]
    rmspe = 100 * np.sqrt(
        np.sum(np.square((signal / original_signal) - 1)) / len(signal)
    )
    rmspes.append(rmspe)

    # diffs = original_specimen - specimen
    # stats.append(np.std(diffs, axis=0))

    ax0.plot(stiff.wavelengths, signal, color=colors[idx], lw=0.4, alpha=1)
    axzoom.plot(stiff.wavelengths, signal, color=colors[idx], lw=0.4, alpha=1)

    print(f"Plot Component {idx+1}")

ax0.plot(
    stiff.wavelengths,
    original_signal,
    label=f"Original Float",
    color="black",
    lw=1.2,
)
axzoom.plot(
    stiff.wavelengths,
    original_signal,
    label=f"Original Float",
    color="black",
    lw=1.2,
)
rmspes.append(
    100
    * np.sqrt(
        np.sum(np.square((original_signal / original_signal) - 1))
        / len(original_signal)
    )
)

ax0.legend(prop={"size": 8}, loc="upper left")
sm = plt.cm.ScalarMappable(
    cmap=cm, norm=plt.Normalize(vmin=0.5, vmax=n_components + 0.5)
)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label(
    "Number of Principal Components",
)
cb.ax.yaxis.labelpad = 8
cb.ax.xaxis.tick_top()
cb.ax.xaxis.set_ticks(list(range(1, n_components + 1)))
cb.ax.xaxis.set_label_position("top")

COLOR_SIZE = "#69b3a2"
COLOR_RMSE = "black"
# quality vs file size
ax1a = ax1.twinx()
ax1.set_xlabel("Signal Format")
ax1.set_ylabel("File Size (MB)", color=COLOR_SIZE)
ax1a.tick_params(axis="y", labelcolor=COLOR_SIZE)
files = np.array(files)
file_size = files[:, 2]
x = [str(r) for r in range(1, n_components + 1)] + ["Original"]
ax1.bar(x, file_size, color=COLOR_SIZE)
ax1.set_yscale("log")

# signal error plot
ax1a.plot(x, rmspes, color=COLOR_RMSE)
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
    "scripts/jpg/pca-stats.pdf",
    bbox_inches="tight",
    transparent=True,
)
fig.savefig(
    "scripts/jpg/pca-stats.svg",
    bbox_inches="tight",
    transparent=True,
)
