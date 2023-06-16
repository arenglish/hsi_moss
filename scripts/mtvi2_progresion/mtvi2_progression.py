import os
from pathlib import Path
import sys
sys.path.append(Path(os.getcwd()).as_posix())
from src.hsi_moss.dataset import *
from src.hsi_moss.raster import *

from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
df = read_csv(basepath.joinpath("moss_copy.csv").as_posix())

dates = [
    '16 August 2018',
    '20 September 2018',
    '7 November 2018',
    '7 December 2018'
]
def plot_sds(d: DataFrame):
    # gridspec inside gridspec
    fig = plt.figure(layout='constrained', dpi=300, figsize=(20,24))
    fig.suptitle(f"Sample {d.iloc[0]['SampleId']}\n(Site - {d.iloc[0]['Site']} | History - {d.iloc[0]['History']} | Species - {d.iloc[0]['Species']} | Treatment - {d.iloc[0]['Treatment']})", fontsize=24)

    gs = gridspec.GridSpec(5, 4, figure=fig)

    # plot mean signals
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance (au)')
    ax1.set_xlim(500,1005)

    # plot mtvi2 progression
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_xlabel('Session #')
    ax2.set_ylabel('MTVI2 (au)')
    ax2.set_xticks([1,2,3,4])
    ax2.set_xlim(0.5,4.5)

    # plot continuum removals
    ax3 = fig.add_subplot(gs[3, :2])
    ax3.set_xlabel('Session #')
    ax3.set_ylabel('Continuum Removal')
    ax3.set_xlim(0.5,4.5)
    ax3.set_xticks([1,2,3,4])

    ax4 = fig.add_subplot(gs[3, 2:])
    ax4.set_xlabel('Session #')
    ax4.set_xlim(0,5)
    ax4.set_xlim(0.5,4.5)
    ax4.set_xticks([1,2,3,4])

    ax5 = fig.add_subplot(gs[4, :2])
    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_ylabel('Cab Continuum Removal')
    ax5.set_xlim(600, 750)
    ax6 = fig.add_subplot(gs[4, 2:])
    ax6.set_xlabel('Wavelength (nm)')
    ax6.set_ylabel('LD Continuum Removal')
    ax6.set_xlim(700, 800)
    isdry = d.iloc[0]['Treatment'] == 'dry'


    for idx, (i,row) in enumerate(d.iterrows()):
        date = dates[row['Session']-1]
        sample_name = f't{row["Session"]}s{row["SampleId"]}'
        specimen_mean_path = DatasetOutput(sample_name, DatasetOutputTypes.specimen_mean, basepath)
        df_mean = read_csv(specimen_mean_path.filepath.as_posix())

        stiff_path = DatasetOutput(sample_name, DatasetOutputTypes.stiff_corrected, basepath).filepath
        rgb = STiff(stiff_path, TiffOptions(rgb_only=True)).rgb
        ax = fig.add_subplot(gs[0, idx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(rgb)

        ax1.plot(df_mean['# wavelengths'].values[36:], df_mean['reflectance'].values[36:], lw=.7, label=date)

        cr_datapath = specimen_mean_path.astype(DatasetOutputTypes.continuum_removal).filepath.with_suffix('.csv')
        cr_data = read_csv(cr_datapath.as_posix())
        ax5.plot(cr_data['cr0_wavelengths'], cr_data['cr0_intensities'], label=date)
        ax5.fill(cr_data['cr0_wavelengths'], cr_data['cr0_intensities'], color='r' if (isdry is True & idx > 1) else 'g', alpha=.15)
        
        ax6.plot(cr_data['cr1_wavelengths'], cr_data['cr1_intensities'], label=date)
        ax6.fill(cr_data['cr1_wavelengths'], cr_data['cr1_intensities'], color='r' if (isdry is True & idx > 1) else 'g', alpha=.15)

        
    d.plot.line('Session','mtvi2_mean',ax=ax2, marker='x', color='black')
    d.plot.line('Session','cr0',ax=ax3, marker='x', color='black')
    d.plot.line('Session','cr1',ax=ax4, marker='x', color='black')

    ax1.legend(loc='upper left')
    ax5.legend(loc='upper left')
    ax6.legend(loc='upper left')
    fig.savefig(f'scripts/mtvi2_progresion/plots/{d.iloc[0]["SampleId"]}.png')
    plt.close(fig)

    

df.loc[df['Type']=='sample'].groupby(['SampleId']).apply(plot_sds)
print('done')