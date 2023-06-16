from .raster import *
from os import listdir
from PIL import Image
import numpy as np
import pickle
from typing import TypeVar, Generic, NamedTuple, Tuple, Callable
import colorbrewer


class Raster:
    stiff: STiff


class Subject:
    rasters: Raster


class Subjects:
    A0: Subject


class HyperSet:
    datapath: Path = None
    output_paths: Dict[FileTypes, Path] = {}
    stiffs: List[Callable[[], STiff]]


D = TypeVar("D")


class OutputType:
    picklefilepath: Path = None
    filetype: FileTypes
    data: Any = None
    renders: List[Tuple[Path, Image.Image]]


class Output(OutputType, Generic[D]):
    data: D = None

    def __init__(
        self,
        original_stiff_path: Path,
        filetype: FileTypes,
        renders: List[Image.Image] = [],
        data: D = None,
        save_data_to_pickle=False,
    ):
        self.renders = []
        self.filetype = filetype
        if data is not None:
            self.data = data

        if data is not None and save_data_to_pickle is True:
            self.picklefilepath = get_output_filepath(
                original_stiff_path, self.filetype
            ).with_suffix(".pkl")

        for idx, render in enumerate(renders):
            render_path = get_output_filepath(
                original_stiff_path, filetype
            ).with_suffix(".png")

            render_path = render_path.with_stem(render_path.stem + f"_render{idx}")
            self.renders.append((render_path, render))

    def write_files(self, use_unique_filenames: bool = False):
        if self.data is not None:
            if isinstance(self.data, STiff):
                filepath = self.data.paths.get_filepath(self.filetype)
                self.data.write(
                    self.data.paths.unique(filepath)
                    if use_unique_filenames
                    else filepath
                )
            elif self.picklefilepath is not None:
                with open(
                    (
                        get_unique_filepath(self.picklefilepath.as_posix())
                        if use_unique_filenames
                        else self.picklefilepath
                    ).as_posix(),
                    "wb",
                ) as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for render in self.renders:
            render[1].save(
                (
                    get_unique_filepath(render[0].as_posix())
                    if use_unique_filenames
                    else render[0]
                ).as_posix()
            )


class Output_Corrected(Output[STiff]):
    def __init__(self, stiff: STiff):
        super().__init__(
            original_stiff_path=stiff.paths.get_stiffpath(),
            filetype=FileTypes.cube_corrected,
            data=stiff,
        )


class Output_MTVI2(Output[Image.Image]):
    def __init__(self, original_stiff_path: Path, mtvi2_image: Image.Image):
        super().__init__(
            original_stiff_path=original_stiff_path,
            filetype=FileTypes.mtvi2,
            renders=[mtvi2_image],
        )


class Output_MaskedPot(Output[STiff]):
    def __init__(self, stiff: STiff):
        super().__init__(
            original_stiff_path=stiff.paths.get_stiffpath(),
            filetype=FileTypes.stiff_maskedPot,
            data=stiff,
        )


class Output_SegmentedKMeans(Output[Tuple]):
    def __init__(self, original_stiff_path: Path, labels_raster: np.ndarray, clusters):
        n_colors = len(clusters)
        palette = [0, 0, 0] + list(
            np.array(colorbrewer.Dark2[n_colors - 1]).astype(np.uint8).flatten()
        )

        image = Image.fromarray(labels_raster, mode="P")
        image.putpalette(palette)
        super().__init__(
            original_stiff_path=original_stiff_path,
            filetype=FileTypes.segmented_kmeans,
            data=(labels_raster, clusters),
            renders=[image],
            save_data_to_pickle=True,
        )


class Outputs:
    corrected: Output_Corrected
    mtvi2: Output_MTVI2
    masked: Output[STiff]
    segmented: Output[Image.Image]


class Pipeline:
    stiff: STiff
    outputs: Outputs = Outputs()
    preview_channels = (650, 550, 450)

    def __init__(self, stiff: STiff, overwrite_outputs=False):
        self.stiff = stiff

    def run(self):
        return (
            self.correct(self.stiff)
            .mtvi2(
                self.outputs.corrected.data.cube,
                self.outputs.corrected.data.wavelengths,
            )
            .mask_pot(self.outputs.corrected.data)
            .segment(self.outputs.corrected.data.cube)
        )

    def correct(self, stiff: STiff):
        cube = RasterOps.normalize(
            RasterOps.correction_whiteref_from_mask(
                stiff.cube, stiff.masks["gray1"], clip=True
            ),
            min=0,
            max=1,
            asinttype=np.uint16,
        )

        rgb = RasterOps.scale_to_dtype(
            RasterOps.get_channels(cube, self.stiff.wavelengths, self.preview_channels),
            np.uint8,
        )
        stiff_corrected = self.stiff.copy()
        stiff_corrected.cube = cube
        stiff_corrected.rgb = rgb
        output = Output_Corrected(stiff=stiff_corrected)
        self.outputs.corrected = output
        output.write_files()

        return self

    def mask_pot(self, stiff: STiff):
        masked = RasterOps.mask(stiff.cube, stiff.masks["pot"], masked_val=0)
        stiff_masked = self.stiff.copy()
        stiff_masked.cube = masked
        stiff_masked.rgb = stiff_masked.render8bit()
        output = Output_MaskedPot(stiff_masked)
        self.outputs.masked = output
        output.write_files()
        return self

    def mtvi2(self, cube, wavelengths):
        mtvi2 = Image.fromarray(
            RasterOps.normalize(
                np.nan_to_num(
                    RasterOps.mtvi2(RasterOps.normalize(cube), wavelengths), nan=0
                ),
                asinttype=np.uint8,
            ),
            mode="L",
        )
        output = Output_MTVI2(self.stiff.paths.get_stiffpath(), mtvi2)
        self.outputs.mtvi2 = output
        output.write_files()
        return self

    def segment(self, raster):
        segmented, clusters = RasterOps.segment_kmeans(raster, k=5, n_samples=5000)
        output = Output_SegmentedKMeans(
            Path(self.stiff.paths.get_stiffpath()), segmented.astype(np.uint8), clusters
        )
        self.outputs.segmented = output
        output.write_files()
        return self


class Moss(HyperSet):
    tiff_compression: bool

    def __init__(self, datapath: str, tiff_options: TiffOptions = TiffOptions()):
        self.datapath = Path(datapath)
        for output in FileTypes:
            self.output[output] = self.datapath.joinpath(output.value[1])
        self.tiff_compression = tiff_compression
        tiff_files = [
            self.datapath.joinpath(f)
            for f in listdir(self.datapath.as_posix())
            if ".tif" in f
        ]
        stiff_files = []
        for i, f in enumerate(tiff_files):
            is_original = True
            for n, s in enumerate(tiff_files):
                if (i != n and s.stem in f.stem) or "t3_s01B" not in f.stem:
                    is_original = False
                    break

            if is_original is True:
                stiff_files.append(f)

        self.stiffs = [
            lambda: STiff(
                f.as_posix(), compression=8 if self.tiff_compression is True else 0
            )
            for f in stiff_files
        ]

    def process_all(self):
        for idx, stiff in enumerate(self.stiffs):
            print(f"Processing stiff #{idx+1}/{len(self.stiffs)}")
            p = Pipeline(stiff())
            p.run()
