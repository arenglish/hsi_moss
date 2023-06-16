import os
from pathlib import Path
import sys
from typing import List

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.dataset import *
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pickle


basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
datapath = basepath.joinpath("moss_with_experimental_data.csv")
data = read_csv(datapath.as_posix())

data = data.loc[data["Exp:Round"].isnull() == False]
data = data.loc[data["History"] == "bon"]


class Datasets(NamedTuple):
    pmax: List = None
    p_biom: List = None
    p_area: List = None
    nitrogen: List = None
    sd_mean: List = None
    cr0_mean: List = None
    cr1_mean: List = None
    mtvi2_mean: List = None
    r_phot: List = None
    r_ir: List = None


pmax = data[["SampleId", "Session", "Exp:Pmax"]]
p_biom = data[["SampleId", "Session", "Exp:P_biome"]]
p_area = data[["SampleId", "Session", "Exp:P_area"]]
nitrogen = data[["SampleId", "Session", "Exp:nitrogen"]]
get_sdmean_csvpath = lambda sampleId, session: DatasetOutput(
    f"t{session}s{sampleId}", DatasetOutputTypes.specimen_mean, basepath
).filepath
sd_mean = DataFrame(
    dict(
        zip(
            *[
                data.index,
                zip(
                    data["SampleId"].values,
                    data["Session"].values,
                    *np.array(
                        [
                            read_csv(
                                get_sdmean_csvpath(r["SampleId"], r["Session"]),
                                delimiter=",",
                            )["reflectance"].values
                            for idx, r in data.iterrows()
                        ]
                    ).T,
                ),
            ]
        )
    )
).transpose()
get_cr_csvpath = lambda sampleId, session: DatasetOutput(
    f"t{session}s{sampleId}", DatasetOutputTypes.continuum_removal, basepath
).filepath.with_suffix(".csv")
cr0_mean = DataFrame(
    dict(
        zip(
            *[
                data.index,
                zip(
                    data["SampleId"].values,
                    data["Session"].values,
                    *np.array(
                        [
                            read_csv(
                                get_cr_csvpath(r["SampleId"], r["Session"]),
                                delimiter=",",
                            )["cr0_intensities"]
                            .dropna()
                            .values
                            for idx, r in data.iterrows()
                        ]
                    ).T,
                ),
            ]
        )
    )
).transpose()
cr1_mean = DataFrame(
    dict(
        zip(
            *[
                data.index,
                zip(
                    data["SampleId"].values,
                    data["Session"].values,
                    *np.array(
                        [
                            read_csv(
                                get_cr_csvpath(r["SampleId"], r["Session"]),
                                delimiter=",",
                            )["cr1_intensities"]
                            .dropna()
                            .values
                            for idx, r in data.iterrows()
                        ]
                    ).T,
                ),
            ]
        )
    )
).transpose()
d = Datasets(
    ["P. Max", pmax],
    ["P. Biom", p_biom],
    ["P. Area", p_area],
    ["Nitrogen", nitrogen],
    ["SD Mean", sd_mean],
    ["CR0 Mean", cr0_mean],
    ["CR1 Mean", cr1_mean],
)


d_combos = [
    [d.pmax, d.p_biom, d.p_area, d.nitrogen],
    [d.sd_mean, d.cr0_mean, d.cr1_mean],
]

for dx in d_combos[1]:
    pickle.dump([dx, d_combos[0]], open(f"scripts/svm/modelpair-{dx[0]}.pk", "wb"))
