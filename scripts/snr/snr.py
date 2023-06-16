import os
from pathlib import Path
import sys
from typing import List
from pandas import read_csv, DataFrame, Series
import numpy as np
import math
from skimage.restoration import estimate_sigma
from scipy.optimize import curve_fit
from matplotlib import colors, cm, scale
from matplotlib import gridspec
import functools
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.spectral_envi import *

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
datapath = basepath.joinpath("02-moss_capture_order.csv")
data = read_csv(datapath.as_posix())

data = data.loc[data["Type"] == "sample"]


def estimate_noise(I):
    H, W = I.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

    return sigma


snrs = []
wavelengths = None
grayvalues = []


def Gauss(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


for idx, d in data.loc[data["Session"] == 1].iterrows():
    name = f't{d["Session"]}s{d["SampleId"]}'

    stiffpath = basepath.joinpath(d["Stiff:RelativePath"])
    mtiffpath = basepath.joinpath(d["Mtiff:RelativePath"])
    stiff = STiff(stiffpath, TiffOptions(), mtiffpath=mtiffpath)
    sample = stiff.cube
    noise = estimate_sigma(sample, average_sigmas=False, channel_axis=-1)
    wavelengths = stiff.wavelengths

    # snr = mean / stdv
    snrs.append(noise)
    gray = stiff.cube[
        # stiff.masks["gray1"]
        np.logical_or(
            stiff.masks["gray1"],
            np.logical_or(stiff.masks["gray2"], stiff.masks["gray3"]),
        )
        > 0
    ]
    # gray = np.reshape(
    #     stiff.cube[stiff.masks["pot"]],
    #     (len(stiff.masks["pot"][stiff.masks["pot"] == True]), -1),
    # )
    mean = gray.mean(axis=0)

    # plot and save gray spatial histogram
    lognorm = colors.make_norm_from_scale(
        functools.partial(scale.LogScale, nonpositive="mask")
    )(colors.Normalize)

    plt.figure(figsize=(8, 3))

    # plt.savefig(
    #     f"scripts/snr/gray_deviation-{name}.pdf",
    #     transparent=True,
    #     bbox_inches="tight",
    # )
    # plt.savefig(
    #     f"scripts/snr/gray_deviation-{name}.png",
    #     transparent=True,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # plot and save gaussian fits
    params = []
    fits = []
    fig = plt.figure(constrained_layout=True, figsize=(11, 12))
    spec = gridspec.GridSpec(
        ncols=7,
        nrows=4,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, 2 / 5],
        height_ratios=[1, 1, 1, 0.7],
        hspace=0,
        wspace=0,
    )
    ax0 = fig.add_subplot(spec[2, 0:3])
    ax0.margins(x=0, y=0)
    ax0.tick_params(direction="in")
    ax0.set(
        xlim=(-150, 150),
        ylim=(10**0, 10**2.6),
        xlabel="Deviation (ADU)",
        ylabel="Frequency",
    )
    ax0.semilogy(base=2)
    ax0.grid(True, which="both", alpha=0.3)
    ax0.set_axisbelow(True)

    ax = fig.add_subplot(spec[2, 3:6])
    ax.margins(x=0, y=0)
    ax.tick_params(direction="in")
    ax.set_yticklabels([])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set(
        # xlim=(-150, 150),
        ylim=(10**0, 10**2.6),
        xlabel="Deviation (ADU)",
    )
    ax.semilogy(base=2)

    ax_mag = fig.add_subplot(spec[3, :6])
    ax_mag.margins(x=0, y=0)
    ax_mag.set_xlabel("Pixel Signal (ADU)")
    ax_mag.set_ylabel("Standard Deviation (ADU)")
    ax_mag.tick_params(direction="in")

    ax_dens = fig.add_subplot(spec[1, :-1])
    ax_dens.set_xlabel("Wavelength, (nm)")
    ax_dens.set_ylabel("Deviation (ADU)")
    ax_dens.margins(x=0, y=0)
    ax_dens.tick_params(direction="in")

    c = ax_dens.hist2d(
        np.tile(stiff.wavelengths, gray.shape[0]),
        np.squeeze(np.reshape(gray, (gray.shape[0] * gray.shape[1], -1)))
        - np.tile(mean, gray.shape[0]),
        norm=lognorm(),
        bins=100,
    )
    # density colorbar
    cax0 = fig.add_subplot(spec[1, -1])
    cax0.margins(x=0, y=0)
    cb = fig.colorbar(c[3], cax=cax0, orientation="vertical")
    cb.set_label("Frequency", rotation=-90)
    cb.ax.yaxis.labelpad = 14

    # gaussian colorbars
    cax = fig.add_subplot(spec[2:, -1])
    cax.margins(x=0, y=0)

    norm = colors.Normalize(
        math.floor(stiff.wavelengths[0]), math.ceil(stiff.wavelengths[-1])
    )
    for i in range(0, len(stiff.wavelengths), 8):
        g = gray[:, i]
        m = g.mean(axis=0)
        centered = g - m
        y, edges = np.histogram(centered, bins=len(np.unique(centered)))
        x = []
        for ie, p in enumerate(edges):
            if ie != len(edges) - 1:
                x.append((edges[ie + 1] + p) / 2)
        x = np.array(x)
        try:
            parameters, covariance = curve_fit(Gauss, x, y)
            params.append([parameters, covariance])
            fit_x = np.linspace(-150, 150, 301).astype(np.int16)
            fit_y = Gauss(
                fit_x,
                parameters[0],
                parameters[1],
                parameters[2],
            )
            fits.append([fit_x, fit_y, i])
        except:
            print("couldnt find a fit")

        c = ax0.scatter(
            x,
            y,
            c=np.repeat(stiff.wavelengths[i], len(x)),
            # alpha=0.2,
            label="data",
            # lw=1,
            s=4,
            cmap="Spectral",
            norm=norm,
        )

    cb = fig.colorbar(c, cax=cax, orientation="vertical")
    cb.set_label("Wavelength (nm)", rotation=-90)
    cb.ax.yaxis.labelpad = 14

    dev = gray.std(axis=0)
    ax_mag.scatter(mean, dev, s=4, c=stiff.wavelengths, cmap="Spectral", norm=norm)
    # plt.legend()
    for i, fit in enumerate(fits):
        ax.plot(
            fit[0],
            fit[1],
            "-",
            c=cm.Spectral(norm(stiff.wavelengths[fit[2]])),
            lw=1,
            # alpha=0.3,
            label=f"band{i}",
        )

    # pixel sample intensity plot
    ax_pix = fig.add_subplot(spec[0, 4:6])
    ax_pix.set_xlabel("x (px)")
    ax_pix.set_xlabel("y (px)")
    ax_pix.tick_params(direction="in")
    ax_pix.margins(x=0, y=0)
    cax = fig.add_subplot(spec[0, -1])
    cax.margins(x=0, y=0)

    # g = np.reshape(gray, (41, 51, -1))
    length = gray.shape[0]
    y = math.floor(math.sqrt(length))
    x = math.floor(length / y)
    g = gray[: x * y, :]
    g = np.reshape(g, (y, x, -1))
    im = ax_pix.imshow(g[:, :, 60], cmap="plasma", aspect="auto")
    cb = fig.colorbar(im, cax=cax, location="right")
    # cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(direction="in")
    cb.set_label("Pixel Signal (ADU)", rotation=-90)
    cb.ax.yaxis.labelpad = 14

    # plot images
    corrected_stiff = STiff(
        basepath.joinpath(f"03-correction/{name}.corrected.darkcorrect.tif"),
        TiffOptions(rgb_only=True),
    )
    ax = fig.add_subplot(spec[0, :2])
    ax.set_xticks([])
    ax.set_yticks([])

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    center = (54, 470)
    image = cv2.circle(corrected_stiff.rgb, center, 20, (255, 0, 0), 2)
    image = cv2.circle(image, (489, 69), 20, (255, 0, 0), 2)
    image = cv2.circle(image, (487, 482), 20, (255, 0, 0), 2)

    ax.imshow(image, aspect="auto")

    ax = fig.add_subplot(spec[0, 2:4])
    ax.set_xticks([])
    ax.set_yticks([])

    # crop = [np.stiff.masks['gray1']]
    offset = 50
    ax.imshow(
        image[
            center[0] - offset : center[0] + offset,
            center[1] - offset : center[1] + offset,
            :,
        ],
        aspect="auto",
    )

    fig.savefig(
        f"scripts/snr/gaussian_fits-{name}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    for k in range(0, gray.shape[-1]):
        if idx == 0:
            grayvalues.append(list(np.random.choice(gray[:, k], 400)))
        else:
            grayvalues[k] = grayvalues[k] + list(np.random.choice(gray[:, k], 40))

    if idx == 0:
        # take gray reference patches and save pixel histograms
        histdf = {}
        for k in range(0, gray.shape[-1]):
            ch = gray[:, k]
            ch = ch - ch.mean()
            count, val = np.histogram(ch, bins=100)
            header = f"band{k} ({wavelengths[k]:.1f} nm)"
            histdf[f"band{k}"] = k
            histdf[f"band{k}-wavelength"] = wavelengths[k]

            ma = np.amax(val)
            mi = np.amin(val)
            histdf[f"band{k}-value"] = val
            histdf[f"band{k}-count"] = count
        DataFrame(dict([(k, Series(v)) for k, v in histdf.items()])).to_csv(
            "scripts/snr/hist-gray.csv"
        )

        # plot with a label
        plt.plot(
            stiff.wavelengths, noise, color=[0.282, 0.427, 0.659], label="specimens"
        )
    else:
        plt.plot(stiff.wavelengths, noise, color=[0.282, 0.427, 0.659])

    print(idx)
snr_mean = np.array(snrs).mean(axis=0)
plt.plot(wavelengths, snr_mean, color="red", label="mean")
plt.xlabel("Wavelength (nm)")
plt.ylabel("SNR")
plt.savefig("scripts/snr/snr.pdf")
print("done")
