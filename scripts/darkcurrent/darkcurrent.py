import os
from pathlib import Path
import sys
import math
from matplotlib import colors

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.dataset import *
from matplotlib import pyplot as plt

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
datapath = basepath.joinpath("02-moss_capture_order.csv")
data = read_csv(datapath.as_posix())

data = data.loc[data["Type"] == "sample"]
data = data.sort_values(by=["Session", "CaptureOrder"])

fig = plt.figure(layout="constrained", dpi=300)
fig.suptitle("Moss Bioindicator SVR Prediction Models")
ax = fig.add_subplot()
darkcurrentdir = basepath.joinpath("01-raw")
darkcurrentcube = []
for idx, row in data.iterrows():
    sample = row["SampleId"]
    session = row["Session"]
    name = f"t{session}s{sample}"

    darkcurrent = np.loadtxt(
        darkcurrentdir.joinpath(f"{name}.darkref.csv"), delimiter=","
    ).astype(np.uint16)
    stiff = STiff(basepath.joinpath(f"02-stiff/{name}.tif"))
    # scale to 8 bit
    # darkcurrent = np.floor(darkcurrent * (2**8 - 1) / (2**12 - 1))
    # im = Image.fromarray(darkcurrent.astype(np.uint8), mode="L")
    # im.show()
    darkcurrentcube.append(darkcurrent)
    # darkcurrent = darkcurrent / (2**12 - 1)

    # im = Image.fromarray(darkcurrent.astype(np.uint8), mode="L")
    # ax.imshow(darkcurrent, vmax=250, vmin=230)
    # plt.show()
    # x = []
    # y = []
    # z = []
    # for idxs, row in enumerate(darkcurrent):
    #     for idxw, dc in enumerate(row):
    #         x.append(idxs)
    #         y.append(idxw)
    #         z.append(dc)
    # ax.scatter(x, y, z, s=0.4)

darkcurrentcube = np.array(darkcurrentcube)
ma = np.amax(darkcurrentcube)
mi = np.amin(darkcurrentcube)

v = darkcurrentcube.var(axis=0)
s = darkcurrentcube.std(axis=0)
m = darkcurrentcube.mean(axis=0)

for im, name, ticks in [
    (s, "scripts/darkcurrent/darkframe.std.pdf", [0.1, 0.4, 0.7, 1]),
    (v, "scripts/darkcurrent/darkframe.var.pdf", [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]),
    (m, "scripts/darkcurrent/darkframe.mean.pdf", [239, 240, 241, 242, 243]),
]:
    plt.rcParams["savefig.pad_inches"] = 0
    fig = plt.figure(figsize=(1.7, 4), layout="tight")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.02], hspace=0, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0], frameon=False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax_cb = fig.add_subplot(gs[0, 1], frameon=False)
    ax_cb.get_xaxis().set_visible(False)
    # ax_cb.get_yaxis().set_visible(False)
    ax_cb.set_xticks([])

    ma = np.amax(im)
    mi = np.amin(im)
    ra = ma - mi
    plt.autoscale(tight=True)

    i = ax1.imshow(im, cmap="inferno")
    plt.colorbar(i, cax=ax_cb, location="right", ticks=ticks)
    ax_cb.set_yticklabels(list(np.array(ticks).astype(str)))
    fig.savefig(name, transparent=True, bbox_inches="tight")
    fig.clear()

stacked = np.vstack([m.flatten(), s.flatten()]).T
rand_idx = np.random.choice(
    np.array(list(range(stacked.shape[0]))), size=math.floor(stacked.shape[0] / 50)
)
stacked = np.sort(stacked[rand_idx, :], axis=0)
np.savetxt(
    "scripts/darkcurrent/darkstats.csv",
    stacked,
    delimiter=",",
    header="mean, standard deviation",
    comments="",
)

plt.close("all")

plt.imshow(darkcurrentcube[:, 270 : 270 + 50, 170 : 170 + 50].mean(axis=0))
plt.show()

print("done")
