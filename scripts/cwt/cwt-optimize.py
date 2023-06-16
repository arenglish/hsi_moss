import os
from pathlib import Path
import sys

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.dataset import *

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
datapath = basepath.joinpath("moss_copy.csv")
data = read_csv(datapath.as_posix())

data = data.loc[data["Type"] == "sample"]

cwt_cube = []
for idx, row in data.iterrows():
    samplename = f"t{row['Session']}s{row['SampleId']}"
    cwtpath = DatasetOutput(samplename, DatasetOutputTypes.cwt, basepath).filepath

    stiff = STiff(cwtpath.as_posix())
    cwt = np.squeeze(stiff.cube)
    cwt_cube.append(cwt)

cwt_cube = np.array(cwt_cube)
