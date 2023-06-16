import os
from pathlib import Path
import sys
from pandas import read_csv, DataFrame

sys.path.append(Path(os.getcwd()).as_posix())

from matplotlib import pyplot as plt

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
datapath = basepath.joinpath("02-moss_capture_order.csv")
data = read_csv(datapath.as_posix())

data = data.loc[data["Type"] == "sample"]
data = data.sort_values(by=["Session", "CaptureOrder"])

wavelengths = []
sds = []
sds_dark = []

for idx, row in data.iterrows():
    sample = row["SampleId"]
    session = row["Session"]
    name = f"t{session}s{sample}"
    df_mean = read_csv(basepath.joinpath(f"03b-specimen_mean/{name}.mean.csv"))
    sd = df_mean["reflectance"].values
    sds.append(sd)
    wavelengths = df_mean["# wavelengths"]
    df_mean_dark = read_csv(
        basepath.joinpath(f"03b-specimen_mean/{name}.mean.darkcorrect.csv")
    )
    sd_dark = df_mean_dark["reflectance"].values
    sds_dark.append(sd_dark)
    diff = sd - sd_dark
    if diff[diff < 0].any():
        print("err")


print("done")
