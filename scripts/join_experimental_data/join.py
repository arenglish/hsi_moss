import os
from pathlib import Path
import sys

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
import numpy as np
from pandas import DataFrame, read_csv

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
dataset = read_csv(basepath.joinpath("moss_copy.csv").as_posix())
experimental_data = read_csv(
    basepath.joinpath("experimental_data/mesocosm_spec_trait.txt"), delimiter=" "
)

dataset["Exp:SpeciesFull"] = np.nan
dataset["Exp:Treatment"] = np.nan
dataset["Exp:Round"] = np.nan
dataset["Exp:Pmax"] = np.nan
dataset["Exp:P_biome"] = np.nan
dataset["Exp:P_area"] = np.nan
dataset["Exp:nitrogen"] = np.nan

for idx, line in experimental_data.iterrows():
    meso: str = line["MESO"]
    meso = meso.rjust(3, "0")
    sample_id = meso
    round = int(line["ROUND"])
    species = line["SPEC"]
    treat = line["TREAT"]
    pmax = line["Pmax"]
    p_biome = line["P_BIOM"]
    p_area = line["P_AREA"]
    nitr = line["N"]

    if treat == "ctr":
        treat = "wet"

    if round == 1:
        session = 2
    elif round == 2:
        session = 3
    else:
        session = 4

    datarow = dataset.loc[
        (dataset["SampleId"] == sample_id) & (dataset["Session"] == session)
    ]
    if len(datarow) == 0:
        print("CANNOT FIND ROW:\n")
        print(line)
    elif len(datarow) > 1:
        print("MATCHED TOO MANY ROWS:\n")
        print(line)
    datarow["Exp:SpeciesFull"] = species
    datarow["Exp:Treatment"] = species
    datarow["Exp:Round"] = int(round)
    datarow["Exp:Pmax"] = pmax
    datarow["Exp:P_biome"] = p_biome
    datarow["Exp:P_area"] = p_area
    datarow["Exp:nitrogen"] = nitr

    dataset.update(datarow)

dataset.to_csv(basepath.joinpath("moss_with_experimental_data.csv"), index=False)
