import os
from pathlib import Path
import sys
sys.path.append(Path(os.getcwd()).as_posix())
from src.hsi_moss.dataset import *
from src.hsi_moss.raster import *
import random

from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
df = read_csv(basepath.joinpath("moss_copy.csv").as_posix())
df['History'] = df['History'].astype('category')
df['Site'] = df['Site'].astype('category')
df['Species'] = df['Species'].astype('category')
df['Treatment'] = df['Treatment'].astype('category')
df['Session'] = df['Session'].astype('category')

dates = [
    '16 August 2018',
    '20 September 2018',
    '7 November 2018',
    '7 December 2018'
]
fig = plt.figure(layout='tight', dpi=300, figsize=(20,12))
gs = gridspec.GridSpec(2,3, figure=fig)

sites = df.loc[df['Type']=='sample'].groupby(['Site'])

markers = ['o','s','p','*','H','^']

for idxs, sk in enumerate([k for k in sites.groups.keys() if len(sites.groups[k]) >0]):
    site = sites.get_group(sk)

    histories = site.groupby('History')

    markersize = 40
    for idxh, hk in enumerate([k for k in histories.groups.keys() if len(histories.groups[k]) > 0]):
        history = histories.get_group(hk)
        ax = fig.add_subplot(gs[idxs, idxh])
        ax.set_title(hk)
        ax.set_xticks([1,2,3,4])
        ax.set_xlabel('Session')

        if idxh == 0:
            ax.set_ylabel(f'{sk} CR (au)')
        else:
            ax.set_ylabel('CR (au)')
        
        species = history.groupby('Species')
        for idxsp, spk in enumerate([k for k in species.groups.keys() if len(species.groups[k]) > 0]):
            specie = species.get_group(spk)
            treatments = specie.groupby('Treatment')
            wet = treatments.get_group('wet')
            dry = treatments.get_group('dry')
            ax.scatter(wet['Session'].apply(lambda x: int(x)+(random.random()*.1 - .05)),wet['mtvi2_mean'], color='black', marker=MarkerStyle(markers[idxsp], fillstyle='full'), linewidth=.5, alpha=.4, s=markersize, label=spk)
            ax.scatter(dry['Session'].apply(lambda x: int(x)+(random.random()*.1 - .05)),dry['mtvi2_mean'], color='black', marker=MarkerStyle(markers[idxsp], fillstyle='none'), linewidth=.5, s=markersize)
        
        ax.legend()

plt.savefig('scripts\mtvi2_progresion\mtvi2')

print('done')