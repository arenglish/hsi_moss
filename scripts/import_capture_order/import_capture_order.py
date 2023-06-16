import os
from pathlib import Path
import sys

sys.path.append(Path(os.getcwd()).as_posix())
from glob import glob

from src.hsi_moss.raster import *
import numpy as np
from pandas import DataFrame, read_csv

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
dataset = read_csv(basepath.joinpath("moss_copy.csv").as_posix())
dataset["CaptureOrder"] = -1
dataset["CaptureOrder"] = dataset["CaptureOrder"].astype("int")

for idx, sample in dataset.loc[dataset["Type"] == "sample"].iterrows():
    sampleId = sample["SampleId"]
    session = sample["Session"]

    rawDir = basepath.joinpath(f"01-raw/session{session} +/{sampleId}")
    png_file_with_capture_order = glob(rawDir.joinpath("2018*.png").as_posix())

    if len(png_file_with_capture_order) != 1:
        continue
    png_file_with_capture_order = png_file_with_capture_order[0]

    capture_order = int(png_file_with_capture_order.split(".")[0].split("_")[-1])
    dataset.at[idx, "CaptureOrder"] = capture_order

dataset.to_csv(basepath.joinpath("02-moss_capture_order.csv"), index=False)
