import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pandas import read_csv
from pathlib import Path
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,6), layout='tight')
gs = fig.add_gridspec(3, 2, height_ratios=[1.2,1,1], width_ratios=[1,.01], hspace=.05, wspace=.01)
ax1 = fig.add_subplot(gs[0, 0])
ax1.tick_params(direction="in")
ax_cb = fig.add_subplot(gs[0, 1])
ax_cb.set_xticks([])
ax_cb.set_yticks([])

ax2 = fig.add_subplot(gs[2, 0])
ax2.tick_params(direction="in")

ax3 = fig.add_subplot(gs[1, 0])
ax3.tick_params(direction="in")


def kern(l, w):
    return signal.morlet2(l,w)

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
sd_mean = read_csv(basepath.joinpath(r"03b-specimen_mean\t3s22A.mean.csv"))
wavelengths = sd_mean['# wavelengths'].values[20:]
sd_mean = sd_mean['reflectance'].values[20:]
# wavelengths_new = np.arange(500,1004)
# sd_mean_interp = np.interp(wavelengths_new, sd_mean['# wavelengths'].values, sd_mean['reflectance'].values)
widths = np.arange(1, 31)
# widths = [1,20,80,100]
cwtmatr = signal.cwt(sd_mean, kern,  widths)
cwtmatr_yflip = np.flipud(np.real(cwtmatr))

ax2.plot(wavelengths, sd_mean, color='black', lw=.6)
ax2.set_ylabel('Reflectance (au)')
ax2.set_xlabel('Wavelength (nm)')
ax2.grid(alpha=.4, which='both')

vmin = 1
vmax = -1
im = ax1.imshow(cwtmatr_yflip, aspect='auto', cmap='bone', vmax=vmax, vmin=vmin, interpolation='nearest')
plt.colorbar(im, cax=ax_cb, pad=0.01, ticks=[-1,0,1])
ax1.set_ylabel('Scale')
ax1.set_xticks([])
# ax1.set_xlabel('Wavelength (nm)')
ax1.set_xticklabels([])


M = len(wavelengths)
s = 4.0
w = 2.0
for idx, width in enumerate(widths):
    wavelet = kern(np.min([10 * width, len(sd_mean)]), width)
    ax3.plot(wavelengths[:len(wavelet)], np.abs(wavelet), lw=.6, label=f'scale {idx}')
# ax3.set_xlabel('Wavelength (nm)')
ax3.grid(alpha=.4, which='both')
ax3.set_xticklabels([])
# ax3.legend(prop={'size': 6})

fig.savefig('scripts\cwt\cwt.png', transparent=True)
plt.show()