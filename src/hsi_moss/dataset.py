from pathlib import Path
from typing import NamedTuple, List, Dict
from os import makedirs
import csv
import numpy as np
from collections import namedtuple
from pandas import DataFrame, read_csv


class DatasetOutputType(NamedTuple):
    name: str
    namespace: str | List[str]
    ext: str
    is_original_source_data: bool = False


class DatasetOutputTypes(NamedTuple):
    raw = DatasetOutputType("01-raw", "", ".raw", True)
    darkreftabular = DatasetOutputType("01-raw", "darkref", ".csv")
    stiff_original = DatasetOutputType("02-stiff", "", ".tif", True)
    stiff_darkcorrected = DatasetOutputType("02-stiff", "darkcorrect", ".tif")
    mtiff = DatasetOutputType("02-stiff", "masks", ".tif")
    rgb = DatasetOutputType("02-stiff", "", ".png")
    stiff_corrected = DatasetOutputType("03-correction", "corrected", ".tif")
    stiff_corrected_dark = DatasetOutputType(
        "03-correction", "corrected.darkcorrect", ".tif"
    )
    stiff_corrected_whiteref = DatasetOutputType(
        "03-correction", "corrected_whitetile", ".tif"
    )
    correction_difference = DatasetOutputType(
        "03-correction", "correction_diff(pot_mean)", ".csv"
    )
    decorrelation = DatasetOutputType("03a-decorrelation", "pca", ".tif")
    decorrelation_jpg = DatasetOutputType("03a-decorrelation", "jpg", ".tif")
    specimen_mean = DatasetOutputType("03b-specimen_mean", "mean", ".csv")
    specimen_mean_darkcorrect = DatasetOutputType(
        "03b-specimen_mean", "mean.darkcorrect", ".csv"
    )
    continuum_removal = DatasetOutputType("04-continuum_removal", "cr", ".tif")
    continuum_removal_preview = DatasetOutputType("04-continuum_removal", "cr", ".png")
    composite_index = DatasetOutputType("04-continuum_removal", "composite", ".png")
    stiff_masked_pot = DatasetOutputType("05-masked", "masked_pot", ".tif")
    segmented_kmeans_mtiff = DatasetOutputType(
        "06-segmentation_kmeans", ["kmeans", "masks"], ".tif"
    )
    segmented_kmeans_stiff = DatasetOutputType(
        "06-segmentation_kmeans", "kmeans", ".tif"
    )
    segmented_kmeans_preview = DatasetOutputType(
        "06-segmentation_kmeans", "segmented_kmeans", ".png"
    )
    segmented_kmeans_reconstructed = DatasetOutputType(
        "06-segmentation_kmeans", "reconstructed", ".tif"
    )
    preview_single_bands = DatasetOutputType("07-unpacked_bands", "band{idx}", ".png")
    mtvi2 = DatasetOutputType("08-mtvi2", "mtvi2", ".tif")
    cwt = DatasetOutputType("09-cwt", "cwt", ".tif")


class MossDatasetCSVKeys(NamedTuple):
    index: str = "Index"
    session: str = "Session"
    sample_id: str = "SampleId"
    type: str = "Type"
    site: str = "Site"
    history: str = "History"
    species: str = "Species"
    treatment: str = "Treatment"
    capture_date: str = "Capture Date"
    rawpath: str = "Raw:RelativePath"
    rawfilename: str = "Raw:Filename"
    stiffpath: str = "Stiff:RelativePath"
    mtiffpath: str = "Mtiff:RelativePath"
    gray1x: str = "gray1:x"
    gray1y: str = "gray1:y"
    gray2x: str = "gray2:x"
    gray2y: str = "gray2:y"
    gray3x: str = "gray3:x"
    gray3y: str = "gray3:y"
    potx: str = "pot:x"
    poty: str = "pot:y"
    pot_radius: str = "pot:radius"
    cornerx: str = "corner:x"
    cornery: str = "corner:y"


SpecimenData = namedtuple("SpecimenData", MossDatasetCSVKeys._fields)


class MossCSV(Dict):
    csvkeys = MossDatasetCSVKeys()
    csvpath: Path
    df: DataFrame
    df_samples: DataFrame
    df_whites: DataFrame

    def __init__(self, csvpath: str | Path):
        self.csvpath = Path(csvpath)
        self.df = read_csv(self.csvpath.as_posix())
        self.df[self.csvkeys.treatment] = self.df[self.csvkeys.treatment].astype(
            "category"
        )
        self.df[self.csvkeys.species] = self.df[self.csvkeys.species].astype("category")
        self.df[self.csvkeys.history] = self.df[self.csvkeys.history].astype("category")
        self.df[self.csvkeys.site] = self.df[self.csvkeys.site].astype("category")

        self.df_samples = self.df.drop(
            (self.df[self.df[self.csvkeys.type] != "sample"]).index
        )
        self.df_whites = self.df.drop(
            (self.df[self.df[self.csvkeys.type] == "sample"]).index
        )

    def save(self):
        self.df.to_csv(self.csvpath.as_posix(), index=False)


class DatasetOutput(NamedTuple):
    datasource_name: str
    type: DatasetOutputType
    basepath: Path

    def mkdirs(self, path: Path = None):
        if path is None:
            path = self.filepath
        if not path.parent.exists():
            makedirs(path.parent)

        return self

    @property
    def filepath(self):
        """Relative filepath of output file"""
        path = Path(self.type.name)
        # if namespace is array of strings, join them with separator
        namespace = (
            self.type.namespace
            if type(self.type.namespace) == str
            else ".".join(self.type.namespace)
        )
        path = path.joinpath(
            ".".join([n for n in [self.datasource_name, namespace] if n != ""])
            + self.type.ext
        )
        return self.basepath.joinpath(path)

    def astype(self, type: DatasetOutputType):
        return DatasetOutput(self.datasource_name, type, self.basepath)


def get_unique_filepath(path: str | Path):
    path = Path(path)

    if not path.exists():
        return path

    unique = False
    count = 0
    newpath = path

    while unique is False:
        count = count + 1
        newpath = newpath.with_stem(f"{path.stem}-{count}")
        if not newpath.exists():
            unique = True

    return newpath


class DatasetPaths:
    BASEPATH: Path
    original_source_safety_flag = True

    def __init__(self, basepath: str | Path):
        self.BASEPATH = Path(basepath)

    def get_outputpath(self, output: DatasetOutput):
        return self.BASEPATH.joinpath(output.filepath)
