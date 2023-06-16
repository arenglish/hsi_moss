from .raster import *
from os import listdir
import numpy as np
from .dataset import *
from .pipeline import *
import PIL.Image as Image
import colorbrewer
from matplotlib import pyplot as plt
from typing import TypedDict, NamedTuple
import csv
import glob
import pickle
from .envi2tiff import *
from .colorspace import *
from .camera_simulation import cam_sim
import pywt


class MossPipelineState(TypedDict):
    original_stiff: STiff
    corrected: np.ndarray
    whiteref: STiff
    correction_differences: List
    stiff_count: int
    stiff_idx: int


class Op_Correction(PipelineOperator[MossPipelineState]):
    def run(self, state, options):
        self.input


class CorrectDiffs(NamedTuple):
    wavelengths: np.ndarray = None
    sd_white: np.ndarray = None
    sd_gray: np.ndarray = None
    name: str = ""


class MossProcessor:
    stiffs: List[Path]
    paths: DatasetPaths
    pipeline_options: PipelineOptions
    tiff_options: TiffOptions
    csvdata: MossCSV

    def run_envi_to_tiff(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        session = input.datasource_name[1]
        sample = input.datasource_name[3:]
        rawpath = Path(
            glob.glob(
                f"{input.filepath.parent.as_posix()}/session*{session}*/{sample}/**/*.raw"
            )[0]
        )
        (cube, wavelengths, metadata) = envi2cube(rawpath)
        darkrawpath = Path(
            glob.glob(
                f"{input.filepath.parent.as_posix()}/session*{session}*/{sample}/**/DARK*.raw"
            )[0]
        )
        (darkcube, darkwavelengths, darkmetadata) = envi2cube(darkrawpath)

        stiff = STiff(output.filepath, self.tiff_options)
        stiff.cube = cube
        stiff.wavelengths = wavelengths
        stiff.metadata = ""
        stiff.render8bit()
        stiff.write_stiff()

        # create dark corrected stiff
        dcube = cube - darkcube
        stiff = STiff(
            output.astype(DatasetOutputTypes.stiff_darkcorrected).filepath,
            self.tiff_options,
        )
        stiff.cube = dcube
        stiff.wavelengths = wavelengths
        stiff.metadata = ""
        stiff.render8bit()
        stiff.write_stiff()
        print("done")

    def run_darkref_tabular(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        session = input.datasource_name[1]
        sample = input.datasource_name[3:]
        darkrawpath = Path(
            glob.glob(
                f"{input.filepath.parent.as_posix()}/session*{session}*/{sample}/**/DARK*.raw"
            )[0]
        )
        (darkcube, darkwavelengths, darkmetadata) = envi2cube(darkrawpath)
        darkcube = np.squeeze(darkcube).astype(np.uint16)
        np.savetxt(output.filepath.as_posix(), darkcube, delimiter=",")
        mi = 239
        ma = 246
        plt.imsave(
            fname=output.filepath.with_suffix(".png").as_posix(),
            arr=darkcube,
            vmax=ma,
            vmin=mi,
            cmap="inferno",
        )

    def run_decorrelation(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(
            input.filepath,
            self.tiff_options,
            mtiffpath=input.astype(DatasetOutputTypes.mtiff).filepath,
        )
        raster, components, mean = RasterOps.cube2pca(
            stiff.cube, ignore_mask=stiff.masks["pot"]
        )

        stiff = stiff.copy()
        stiff.cube = raster.astype(np.float32)
        stiff.type = STIFF_TYPE.PCA
        stiff.extras = {
            "components": components,
            "mean": np.array([mean]),
            "wavelengths": stiff.wavelengths,
        }
        stiff.filepath = output.filepath
        num_components = len(components)
        stiff.wavelengths = np.array(list(range(num_components)), dtype=np.float32)
        stiff.render8bit()
        stiff.write_stiff()

    def run_specimen_mean(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        _input: DatasetOutput,
        _output: DatasetOutput,
    ):
        def pr(input: DatasetOutput, output: DatasetOutput):
            stiff = STiff(
                input.filepath,
                self.tiff_options,
                input.astype(DatasetOutputTypes.mtiff).filepath,
            )
            masked = stiff.cube[stiff.masks["pot"] > 0]
            mean = np.mean(masked, axis=0)

            X = np.concatenate(
                [stiff.wavelengths[:, np.newaxis], mean[:, np.newaxis]], axis=-1
            )

            np.savetxt(
                output.mkdirs().filepath,
                X,
                delimiter=",",
                header="wavelengths,reflectance",
            )

        pr(_input, _output)
        pr(
            _input.astype(DatasetOutputTypes.stiff_corrected_dark),
            _output.astype(DatasetOutputTypes.specimen_mean_darkcorrect),
        )

    def run_correct(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read(
            read_masks=input.astype(DatasetOutputTypes.mtiff).filepath
        )

        corrected = RasterOps.correction_whiteref_from_mask(
            stiff.cube, stiff.masks["gray1"], clip=True
        )

        state["corrected"] = corrected

        rstiff: STiff = stiff.copy()
        rstiff.filepath = output.filepath
        rstiff.cube = corrected
        rstiff.render8bit()
        rstiff.write_stiff()

        # darkcorrected
        dstiff = STiff(
            input.astype(DatasetOutputTypes.stiff_darkcorrected).filepath,
            self.tiff_options,
        ).read(read_masks=input.astype(DatasetOutputTypes.mtiff).filepath)

        dcorrected = RasterOps.correction_whiteref_from_mask(
            dstiff.cube, dstiff.masks["gray1"], clip=True
        )

        state["corrected"] = dcorrected

        dstiff: STiff = dstiff.copy()
        dstiff.filepath = output.astype(
            DatasetOutputTypes.stiff_corrected_dark
        ).filepath
        dstiff.cube = dcorrected
        dstiff.render8bit()
        dstiff.write_stiff()

    def run_correct_whitetile(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read(
            read_masks=input.astype(DatasetOutputTypes.mtiff).filepath
        )
        stiff_whiteref = state["whiteref"]

        corrected = RasterOps.correction_from_raster_with_mask(
            stiff.cube, stiff_whiteref.cube, stiff_whiteref.masks["whiteref"], clip=True
        )
        state["corrected"] = corrected

        if options.write_outputs is True:
            stiff: STiff = stiff.copy()
            stiff.filepath = output.filepath
            stiff.cube = corrected
            stiff.render8bit()
            stiff.write_stiff()

    def run_correction_difference(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        if "correction_differences" not in state.keys():
            state["correction_differences"] = []

        correction_gray = STiff(
            input.astype(DatasetOutputTypes.stiff_corrected).filepath,
            TiffOptions(0, False),
        ).read(read_masks=input.astype(DatasetOutputTypes.mtiff).filepath)
        correction_white = STiff(
            input.astype(DatasetOutputTypes.stiff_corrected_whiteref).filepath,
            TiffOptions(0, False),
        ).read(read_masks=input.astype(DatasetOutputTypes.mtiff).filepath)

        correction_gray_sd = RasterOps.to_pixel(
            correction_gray.cube[correction_gray.masks["pot"] > 0]
        )
        correction_white_sd = RasterOps.to_pixel(
            correction_white.cube[correction_gray.masks["pot"] > 0]
        )
        X = np.zeros((len(correction_gray_sd), 3), dtype=object)
        X[:, 0] = correction_gray.wavelengths
        X[:, 1] = correction_gray_sd
        X[:, 2] = correction_white_sd
        np.savetxt(
            output.mkdirs().filepath,
            X,
            delimiter=",",
            header="wavelength,corrected_gray,corrected_white",
        )
        fig = plt.figure(figsize=(8, 6), dpi=300)
        fig.add_subplot()
        ax = fig.axes[0]
        ax.plot(correction_gray.wavelengths, correction_gray_sd, label="Gray Corrected")
        ax.plot(
            correction_white.wavelengths, correction_white_sd, label="White Corrected"
        )
        ax.legend()
        fig.savefig(output.filepath.with_suffix(".png").as_posix())

        accumulatedplotfile = output.filepath.with_stem(
            "accumulated-sd-plot"
        ).with_suffix(".png")

        state["correction_differences"].append(
            CorrectDiffs(
                correction_gray.wavelengths,
                correction_gray_sd,
                correction_white_sd,
                state["original_stiff"].filepath.name,
            )
        )

        cds: List[CorrectDiffs] = state["correction_differences"]

        # write accumulated sds to csv
        header = "wavelength"
        X = np.zeros((len(correction_gray_sd), len(cds) * 2 + 1), dtype=object)
        for idx, cd in enumerate(cds):
            header = f"{header},{cd.name}_gray,{cd.name}_white"
            X[:, (idx * 2 + 1)] = cd.sd_gray
            X[:, (idx * 2 + 2)] = cd.sd_white
        np.savetxt(
            accumulatedplotfile.with_suffix(".csv"),
            X,
            delimiter=",",
            header=header,
        )

        if state["stiff_idx"] + 1 == state["stiff_count"]:
            fig = plt.figure(figsize=(8, 6), dpi=300)
            fig.add_subplot()
            ax = fig.axes[0]
            X = np.zeros((len(correction_gray_sd), (len(cds) * 2) + 1), dtype=object)
            X[:, 0] = correction_gray.wavelengths
            for idx, cd in enumerate(cds):
                color1 = [1, 0, 0]
                color2 = [0, 0, 1]
                opacity = 0.3
                linewidth = 0.5
                ax.plot(
                    cd.wavelengths,
                    cd.sd_gray,
                    color=color1,
                    alpha=opacity,
                    label="Gray Corrected",
                    linewidth=linewidth,
                ) if idx == 0 else ax.plot(
                    cd.wavelengths,
                    cd.sd_gray,
                    color=color1,
                    alpha=opacity,
                    linewidth=linewidth,
                )
                ax.plot(
                    cd.wavelengths,
                    cd.sd_white,
                    color=color2,
                    alpha=opacity,
                    label="White Corrected",
                    linewidth=linewidth,
                ) if idx == 0 else ax.plot(
                    cd.wavelengths,
                    cd.sd_white,
                    color=color2,
                    alpha=opacity,
                    linewidth=linewidth,
                )
            ax.legend()
            fig.savefig(accumulatedplotfile.as_posix())

    def continuum_removal(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read(
            read_masks=input.astype(DatasetOutputTypes.mtiff).filepath
        )

        idxs = [
            [
                RasterOps.band_idx(stiff.wavelengths, 650),
                RasterOps.band_idx(stiff.wavelengths, 715),
            ],
            [
                RasterOps.band_idx(stiff.wavelengths, 710),
                RasterOps.band_idx(stiff.wavelengths, 780),
            ],
        ]
        wavelengths = np.delete(
            stiff.wavelengths,
            list(range(0, idxs[0][0]))
            + list(range(idxs[1][1] + 1, len(stiff.wavelengths))),
            axis=-1,
        )

        crs = [
            RasterOps.continuum_removal(
                np.delete(
                    stiff.cube,
                    list(range(0, c[0]))
                    + list(range(c[1] + 1, len(stiff.wavelengths))),
                    axis=-1,
                ),
                np.delete(
                    stiff.wavelengths,
                    list(range(0, c[0]))
                    + list(range(c[1] + 1, len(stiff.wavelengths))),
                ),
            )
            for c in idxs
        ]

        if options.write_outputs is True:
            X = DataFrame({"wavelength": stiff.wavelengths})
            # get pot spatial mean
            sd_mean = RasterOps.to_pixel(stiff.cube[stiff.masks["pot"] > 0]).flatten()
            X["sd"] = sd_mean
            for idx, (cr, wavelengths) in enumerate(crs):
                stiffcr: STiff = STiff(
                    output.filepath.with_stem(output.filepath.stem + str(idx)),
                    self.tiff_options,
                )
                stiffcr.cube = cr
                stiffcr.wavelengths = wavelengths
                stiffcr.render8bit(channels=(710, 660, 720))
                stiffcr.write_stiff()

                # get continuum removal spatial mean
                cr_mean = RasterOps.to_pixel(
                    stiffcr.cube[stiff.masks["pot"] > 0]
                ).flatten()

                # cr end bands
                X[f"cr{idx}_ends_wavelengths"] = np.NaN
                X.loc[:1, f"cr{idx}_ends_wavelengths"] = [
                    wavelengths[0],
                    wavelengths[-1],
                ]
                # cr end intensities
                X[f"cr{idx}_ends_intensity"] = np.NaN
                X.loc[:1, f"cr{idx}_ends_intensity"] = [
                    sd_mean[idxs[idx][0]],
                    sd_mean[idxs[idx][1]],
                ]

                # all cr bands
                X[f"cr{idx}_wavelengths"] = np.NaN
                X.loc[: len(wavelengths) - 1, f"cr{idx}_wavelengths"] = wavelengths

                # cr spatial mean
                X[f"cr{idx}_intensities"] = np.NaN
                X.loc[: len(cr_mean) - 1, f"cr{idx}_intensities"] = cr_mean

                # cr sum
                X[f"cr{idx}_sum"] = np.NaN
                cr_mean = np.mean(cr_mean)
                X.loc[0, f"cr{idx}_sum"] = cr_mean

                if f"cr{idx}" not in self.csvdata.df:
                    self.csvdata.df[f"cr{idx}"] = np.NaN
                self.csvdata.df.loc[
                    (
                        self.csvdata.df[self.csvdata.csvkeys.sample_id]
                        == input.datasource_name[3:6]
                    )
                    & (
                        self.csvdata.df[self.csvdata.csvkeys.session]
                        == int(input.datasource_name[1])
                    ),
                    f"cr{idx}",
                ] = cr_mean

            self.csvdata.df.to_csv(self.csvdata.csvpath.as_posix(), index=False)
            X.to_csv(
                output.mkdirs().filepath.with_suffix(".csv").as_posix(), index=False
            )

    def continuum_removal_preview(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiffs = glob.glob(
            input.filepath.with_stem(input.filepath.stem + "*").as_posix()
        )

        for s in stiffs:
            path = Path(s)
            stiff = STiff(path, self.tiff_options).read()
            max_val = stiff.cube.shape[-1]
            intensity_raster = RasterOps.normalize(
                RasterOps.integrate(stiff.cube), min=0, max=max_val, asinttype=np.uint8
            )

            if options.write_outputs is True:
                Image.fromarray(intensity_raster).save(
                    output.filepath.with_stem(path.stem)
                )

    def composite_index(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff_cab = STiff(input.filepath.with_stem(input.filepath.stem + "0"))
        stiff_ld = STiff(input.filepath.with_stem(input.filepath.stem + "1"))
        max_val = stiff_cab.cube.shape[-1] / 2
        intensity_cab = RasterOps.normalize(
            RasterOps.integrate(stiff_cab.cube), min=0, max=max_val
        )
        intensity_ld = RasterOps.normalize(
            RasterOps.integrate(stiff_ld.cube), min=0, max=max_val
        )

        cab_threshold = 0.2
        ld_threshold = 0.05
        intensity_cab[intensity_cab < cab_threshold] = 0
        intensity_ld[intensity_ld < ld_threshold] = 0

        composite = np.concatenate(
            [intensity_cab[:, :, np.newaxis], intensity_ld[:, :, np.newaxis]], axis=-1
        )
        composite = np.mean(composite, axis=-1)
        composite[intensity_cab == 0] = 0
        composite[intensity_ld == 0] = 0
        cm = plt.get_cmap("RdYlGn")
        composite = cm(composite)

        print("done")

    def run_rgb_preview(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, TiffOptions(0, False)).read()
        sensitivities = cam_sim.sensitivities
        camRGB2XYZ = cam_sim.camRGB2XYZ_specimen
        illuminants = CIE_light_sources()
        D65 = RasterOps.normalize(illuminants[:, 4])
        wavelengths_ill = illuminants[:, 0]
        wavelengths_s = sensitivities[:, 0]
        sensitivities = RasterOps.normalize(sensitivities[:, 1:])

        # interpolate illuminant down to match sensitivity samples
        illuminant = np.interp(wavelengths_s, wavelengths_ill, D65)

        raster = RasterOps.linear_interpolate(
            stiff.cube, stiff.wavelengths, wavelengths_s
        )
        render = RasterOps.render_with_sensitivities(
            reflectance=raster, illuminant_spd=illuminant, sensitivities=sensitivities
        )
        render_shape = render.shape
        render = np.reshape(render, (-1, 3))
        render = np.hstack([np.ones(len(render))[:, np.newaxis], render])
        renderXYZ = np.reshape(render @ camRGB2XYZ, render_shape)

        renderRGB = (np.squeeze(XYZ2RGB(renderXYZ)) * (2**8 - 1)).astype(np.uint8)
        Image.fromarray(stiff.rgb).save(output.mkdirs().filepath)

    def run_masking(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read(
            read_masks=input.astype(DatasetOutputTypes.mtiff).filepath
        )
        masked = RasterOps.mask(stiff.cube, stiff.masks["pot"], masked_val=0)

        if options.write_outputs:
            stiff_masked = stiff.copy()
            stiff_masked.filepath = output.filepath
            stiff_masked.cube = masked
            stiff_masked.render8bit()
            stiff_masked.write_stiff()

    def run_mtvi2(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read(
            read_masks=input.astype(DatasetOutputTypes.mtiff).filepath
        )
        mtvi2 = RasterOps.mtvi2(stiff.cube, stiff.wavelengths)
        mtvi2_rgb = np.nan_to_num(mtvi2, nan=0)
        mtvi2_specimen_mean = np.mean(mtvi2[stiff.masks["pot"] > 0])

        # Get the color map by name:
        cm = plt.get_cmap("viridis")

        mtvi2_rgb = cm(mtvi2_rgb)

        # save mtvi2 data as tif with rgb preview
        stiff = stiff.copy()
        stiff.filepath = output.mkdirs().filepath
        stiff.cube = mtvi2[:, :, np.newaxis]
        stiff.rgb = mtvi2_rgb
        stiff.wavelengths = np.array([0])
        stiff.write_stiff()

        # save specimen mtvi2 mean in database csv
        key = "mtvi2_mean"
        if key not in self.csvdata.df:
            self.csvdata.df[key] = np.NaN
        self.csvdata.df.loc[
            (
                self.csvdata.df[self.csvdata.csvkeys.sample_id]
                == input.datasource_name[3:6]
            )
            & (
                self.csvdata.df[self.csvdata.csvkeys.session]
                == int(input.datasource_name[1])
            ),
            key,
        ] = mtvi2_specimen_mean

    def run_segment_kmeans(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        mtiffpath = input.astype(DatasetOutputTypes.mtiff).filepath
        stiff = STiff(input.filepath, self.tiff_options).read(read_masks=mtiffpath)
        cube = stiff.cube[stiff.masks["pot"] == 1]
        n = 3
        segmented, clusters = RasterOps.segment_kmeans(cube, k=n**2, n_samples=5000)

        if options.write_outputs:
            segmented_reshaped = np.zeros(
                (stiff.cube.shape[0], stiff.cube.shape[1]), dtype=np.uint16
            )
            segmented_reshaped[stiff.masks["pot"] == 1] = segmented + 1
            masks = {}
            u = np.unique(segmented_reshaped)[1:]
            for idx, s in enumerate(u):
                mask = np.zeros((stiff.cube.shape[0], stiff.cube.shape[1]), dtype=bool)
                mask[:] = 0
                mask[segmented_reshaped == s] = 1
                masks["kmeans_" + str(idx).rjust(3, "0")] = mask

            filepath = output.filepath.with_stem(output.filepath.stem + f"{n**2}")
            output.mkdirs(filepath)
            write_mtiff(filepath, masks)

            # create spatial image representation of classes
            classcube = np.zeros((n**2, len(stiff.wavelengths)), dtype=np.float32)
            classcube[:, :] = clusters
            classcube = np.reshape(classcube, (n, n, len(stiff.wavelengths)))
            stiff = stiff.copy()
            stiff.filepath = output.astype(
                DatasetOutputTypes.segmented_kmeans_stiff
            ).filepath
            stiff.cube = classcube
            stiff.render8bit()
            stiff.write_stiff()

    def run_preview_masks(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        mask_files = glob.glob(
            input.filepath.with_stem(input.filepath.stem + "*").as_posix()
        )
        for m in mask_files:
            masks = list(read_mtiff(m).values())

            merged = RasterOps.merge_masks_distinct(masks)
            n_colors = len(masks) + 1
            palette = [0, 0, 0] + list(
                np.array(colorbrewer.Greens[n_colors - 1]).astype(np.uint8).flatten()
            )

            image = Image.fromarray(merged, mode="P")
            image.putpalette(palette)
            image.save(Path(m).with_suffix(output.filepath.suffix).as_posix())

    def run_cwt(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        df = read_csv(input.filepath.as_posix())
        wavelet = "morl"  # wavelet type: morlet
        widths = np.arange(1, 64, 4)  # scales for morlet wavelet

        # Compute continuous wavelet transform of the audio numpy array
        wavelet_coeffs, freqs = pywt.cwt(df["reflectance"], widths, wavelet=wavelet)

        stiff = STiff(output.filepath, TiffOptions())
        stiff.cube = wavelet_coeffs[:, :, np.newaxis]
        cm = plt.get_cmap("viridis")
        im = cm(wavelet_coeffs)
        im = (im * 255).astype(np.uint8)
        stiff.rgb = im
        stiff.wavelengths = np.array([0])
        stiff.metadata = ""
        stiff.write_stiff()

    def run_preview_single_bands(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(input.filepath, self.tiff_options).read()
        for b in range(0, stiff.cube.shape[-1]):
            filepath = output.filepath.with_stem(
                output.filepath.stem.replace("{idx}", str(b).rjust(3, "0"))
            )
            output.mkdirs(filepath)

            im = stiff.cube[:, :, b]
            mask = 255 * np.ones_like(im, dtype=np.uint8)
            for y in range(0, int(2 * mask.shape[0] / 3)):
                for x in range(int((mask.shape[1] / 3)), mask.shape[0]):
                    mask[y, x] = 0

            cm = plt.get_cmap("viridis")
            im = cm(im)
            im = Image.fromarray((im * 255).astype(np.uint8))
            if b + 1 < 51 or (b + 1) % 51 != 0:
                mask = Image.fromarray(mask).convert("L")
                im.putalpha(mask)
            else:
                print("not applying mask")
            im.save(filepath.as_posix())

    def jpg(
        self,
        state: MossPipelineState,
        options: PipelineOptions,
        input: DatasetOutput,
        output: DatasetOutput,
    ):
        stiff = STiff(
            input.astype(DatasetOutputTypes.stiff_corrected).filepath, self.tiff_options
        ).read()

        stiff.tiff_options = TiffOptions(
            compression_mode=("jpeg", 5), rgb_only=self.tiff_options.rgb_only
        )
        stiff.cube = (
            RasterOps.normalize(stiff.cube, min=0, max=1) * 2**12 - 1
        ).astype(np.uint16)
        stiff.filepath = input.astype(DatasetOutputTypes.decorrelation_jpg).filepath
        stiff.write_stiff()

    def __init__(self, sourcepipelinedir: str | Path, dist_path: str | Path = None):
        self.paths = DatasetPaths(dist_path)
        self.pipeline_options = PipelineOptions()
        whiterefpath = Path(
            r"I:\moss_data\Austin moss 2023\Moss\pipeline\stiff_original\t2swhite387.tif"
        )
        self.state = MossPipelineState(
            whiteref=STiff(
                whiterefpath,
                TiffOptions(),
                mtiffpath=whiterefpath.with_stem(whiterefpath.stem + ".masks"),
            )
        )

        stiffpaths = self.collect_stiffs_csv("moss_copy.csv")
        self.stiffs = np.array(stiffpaths)
        self.tiff_options = TiffOptions(0, False)

    def collect_stiffs_csv(self, csvpath):
        mosscsv = MossCSV(self.paths.BASEPATH.joinpath(csvpath))
        self.csvdata = mosscsv
        stiff_paths = mosscsv.df_samples[mosscsv.csvkeys.stiffpath]
        return [self.paths.BASEPATH.joinpath(p) for p in stiff_paths]

    def from_seg(self, stiff_path):
        path = Path(stiff_path)
        path = DatasetOutput(
            path.stem,
            DatasetOutputTypes.segmented_kmeans_stiff,
            self.paths.BASEPATH,
        )
        mtiffpath = path.astype(DatasetOutputTypes.segmented_kmeans_mtiff)
        stiff_seg = STiff(
            path.filepath,
            TiffOptions(),
            mtiffpath=mtiffpath.filepath.with_stem(mtiffpath.filepath.stem + "256"),
        ).read()

        sample_mask = stiff_seg.masks[list(stiff_seg.masks.keys())[0]]

        reconstructed = stiff_seg.copy()
        reconstructed.cube = np.zeros(
            (sample_mask.shape[0], sample_mask.shape[1], stiff_seg.cube.shape[-1]),
            dtype=np.float32,
        )
        reconstructed.filepath = mtiffpath.astype(
            DatasetOutputTypes.segmented_kmeans_reconstructed
        ).filepath

        mask_names = sorted(stiff_seg.masks.keys())
        for idx, c in enumerate(
            np.reshape(stiff_seg.cube, (-1, stiff_seg.cube.shape[-1]))
        ):
            mask = stiff_seg.masks[mask_names[idx]]
            reconstructed.cube[mask > 0] = c

        reconstructed.render8bit()
        reconstructed.write_stiff()

    class STEPS(Enum):
        correction_grayref = 0
        correction_whiteref = 1
        correction_difference = 2
        decorrelation = 3
        envi2tiff = 4
        specimen_mean = 5
        single_band_previews = 6
        continuum_removal = 7
        continuum_removal_preview = 8
        composite_index = 9
        rgb_preview = 10
        mask_pot = 11
        segment_kmeans = 12
        preview_segmentation_kmeans = 13
        mtvi2 = 14
        cwt = 15
        darkreftabular = 16
        jpg = 17

    def process(self, range=None, steps=None, overwrite=False, skip=[]):
        stiffs = self.stiffs if range is None else self.stiffs[range]
        self.state["stiff_count"] = len(stiffs)
        for idx, path in enumerate(stiffs):
            self.state["stiff_idx"] = idx
            try:
                mtiffpath = DatasetOutput(
                    path.stem, DatasetOutputTypes.mtiff, self.paths.BASEPATH
                ).filepath
                stiff = STiff(
                    path.as_posix(),
                    tiff_options=self.tiff_options,
                    mtiffpath=mtiffpath,
                )
            except Exception as e:
                log.error(e)
                continue
            print(f"Processing image: {idx}/{len(stiffs)}")

            pipeline: Pipeline[MossPipelineState] = Pipeline(self.pipeline_options)
            self.state.update(MossPipelineState(original_stiff=stiff))
            pipeline.initialize_state(self.state)
            output = DatasetOutput(
                stiff.filepath.stem,
                DatasetOutputTypes.stiff_original,
                self.paths.BASEPATH,
            )
            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.correction_grayref.name,
                input=output,
                output=output.astype(DatasetOutputTypes.stiff_corrected),
                run=self.run_correct,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.correction_whiteref.name,
                input=output.astype(DatasetOutputTypes.stiff_original),
                output=output.astype(DatasetOutputTypes.stiff_corrected_whiteref),
                run=self.run_correct_whitetile,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.envi2tiff.name,
                input=output.astype(DatasetOutputTypes.raw),
                output=output.astype(DatasetOutputTypes.stiff_original),
                run=self.run_envi_to_tiff,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.darkreftabular.name,
                input=output.astype(DatasetOutputTypes.raw),
                output=output.astype(DatasetOutputTypes.darkreftabular),
                run=self.run_darkref_tabular,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.correction_difference.name,
                input=output.astype(DatasetOutputTypes.stiff_original),
                output=output.astype(DatasetOutputTypes.correction_difference),
                run=self.run_correction_difference,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.decorrelation.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.decorrelation),
                run=self.run_decorrelation,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.specimen_mean.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected_whiteref),
                output=output.astype(DatasetOutputTypes.specimen_mean),
                run=self.run_specimen_mean,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.single_band_previews.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.preview_single_bands),
                run=self.run_preview_single_bands,
                skip_if_exists=True,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.continuum_removal.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.continuum_removal),
                run=self.continuum_removal,
                skip_if_exists=True,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.continuum_removal_preview.name,
                input=output.astype(DatasetOutputTypes.continuum_removal),
                output=output.astype(DatasetOutputTypes.continuum_removal_preview),
                run=self.continuum_removal_preview,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op: PipelineOperator[MossPipelineState] = PipelineOperator(
                self.STEPS.composite_index.name,
                input=output.astype(DatasetOutputTypes.continuum_removal),
                output=output.astype(DatasetOutputTypes.composite_index),
                run=self.composite_index,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.rgb_preview.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected_whiteref),
                output=output.astype(DatasetOutputTypes.rgb),
                run=self.run_rgb_preview,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.mask_pot.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.stiff_masked_pot),
                run=self.run_masking,
                skip_if_exists=True,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.segment_kmeans.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.segmented_kmeans_mtiff),
                run=self.run_segment_kmeans,
                skip_if_exists=True,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.preview_segmentation_kmeans.name,
                input=output.astype(DatasetOutputTypes.segmented_kmeans_mtiff),
                output=output.astype(DatasetOutputTypes.segmented_kmeans_preview),
                run=self.run_preview_masks,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.mtvi2.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.mtvi2),
                run=self.run_mtvi2,
                skip_if_exists=True,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.cwt.name,
                input=output.astype(DatasetOutputTypes.specimen_mean),
                output=output.astype(DatasetOutputTypes.cwt),
                run=self.run_cwt,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            op = PipelineOperator(
                self.STEPS.jpg.name,
                input=output.astype(DatasetOutputTypes.stiff_corrected),
                output=output.astype(DatasetOutputTypes.decorrelation_jpg),
                run=self.jpg,
                skip_if_exists=False,
            )
            pipeline.register_operator(op)

            if steps is not None:
                steps = [s for s in steps if s not in skip]
            else:
                steps = [
                    s
                    for s in [st.name for st in self.STEPS.__members__.items()]
                    if s not in skip
                ]

            pipeline.run(_ops=steps, overwrite_existing_outputs=overwrite)
        self.csvdata.df.to_csv(self.csvdata.csvpath.as_posix(), index=False)
