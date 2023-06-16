from dataset import *

csvpath = r"I:\moss_data\Austin moss 2023\Moss\pipeline\moss.csv"

mosscsv = MossCSV(csvpath)


def tr1(d, row):
    rawfilepath = row[mosscsv.col_map.index(mosscsv.csvkeys.rawpath)]
    rawfilepath = "raws\\" + rawfilepath
    return rawfilepath


def tr2(stiffpath, row):
    rawfilepath = Path(row[mosscsv.col_map.index(mosscsv.csvkeys.rawpath)])
    trial = rawfilepath.parent.parent.stem[7]
    sample = rawfilepath.parent.stem[:3]
    name = f"t{trial}s{sample}.tif"
    stiffpath = Path("stiff_original").joinpath(name)
    return stiffpath.as_posix()


def tr3(mtiffpath, row):
    stiffpath = Path(row[mosscsv.col_map.index(mosscsv.csvkeys.stiffpath)])
    mtiffpath = stiffpath.with_stem(stiffpath.stem + ".masks")
    return mtiffpath.as_posix()


mosscsv.transform(mosscsv.csvkeys.rawpath, tr1)
mosscsv.transform(mosscsv.csvkeys.stiffpath, tr2)
mosscsv.transform(mosscsv.csvkeys.mtiffpath, tr3)
mosscsv.save()
