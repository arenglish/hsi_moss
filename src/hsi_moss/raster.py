import numpy as np
from typing import List, Any, Dict, Callable, Generic, TypeVar
from .spectral_tiffs import *
from pathlib import Path
from enum import Enum
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from .utils.logger import log
from copy import copy
from typing import NamedTuple, Tuple
from os import makedirs
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import ast


class RasterOps:
    intermediary_dtype = np.float32

    # @staticmethod
    # def masks_as_image(masks: List[np.ndarray], colors: Tuple[Tuple[int,int,int]]=None, unmasked_color=None):
    #     merged = RasterOps.merge_masks_distinct(masks)
    #     im = Image.

    @staticmethod
    def merge_masks_distinct(masks: List[np.ndarray], unmasked_val=0):
        im = np.zeros_like(masks[0], dtype=np.uint8)
        im.fill(unmasked_val)
        for idx, m in enumerate(masks):
            im[m > 0] = idx + 1

        return im

    @staticmethod
    def segment_kmeans(raster, k, n_samples):
        h = None
        raster = gaussian_filter(raster, sigma=1.5)
        if len(raster.shape) == 3:
            h, w, d = raster.shape
            raster = np.reshape(raster, (h * w, d))

        samples = shuffle(raster, random_state=0, n_samples=n_samples)
        kmeans: KMeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(
            samples
        )
        labels = kmeans.predict(raster)
        if h is not None:
            labels = np.reshape(labels, (h, w, -1))
        return (np.squeeze(labels), kmeans.cluster_centers_)

    @staticmethod
    def normalize(raster, min=None, max=None, asinttype=None):
        max = max if max is not None else np.amax(raster)
        min = min if min is not None else np.amin(raster)

        raster = raster.astype(RasterOps.intermediary_dtype)

        normalized = (raster - min) / (max - min)

        if asinttype is None:
            return normalized

        max = np.iinfo(asinttype).max

        return (normalized * max).astype(asinttype)

    @staticmethod
    def scale_to_dtype(raster, dtype):
        """
        Take current dtype (some int) and scale to another dtype (some int).
        """
        min1 = np.iinfo(raster.dtype).min
        max1 = np.iinfo(raster.dtype).max
        min2 = np.iinfo(dtype).min
        max2 = np.iinfo(dtype).max

        norm = (raster - min1) / (max1 - min1)

        return (norm * (max2 - min2) + min2).astype(dtype)

    @staticmethod
    def histogram_clip(raster, from_pct=0.5, range=(0, 1), bins=256):
        hist = np.histogram(raster, bins=256, range=range())
        total = raster.size
        count = 0
        i = 0

        while count < total * 0.5:
            count += hist[0][i]
            i += 1

        max_intensity = hist[1][i]
        raster[raster > max_intensity] = 0
        return RasterOps.normalize(raster)

    @staticmethod
    def to_pixel(raster):
        return np.mean(
            raster,
            axis=tuple(range(0, len(raster.shape) - 1)),
            dtype=RasterOps.intermediary_dtype,
        )

    @staticmethod
    def to_shape(raster, shape):
        return np.broadcast_to(raster, shape)

    @staticmethod
    def band_idx(wavelengths, wavelength):
        return (np.abs(wavelengths - wavelength)).argmin()

    @staticmethod
    def correction_whiteref(raster, whiteref_raster, clip=False):
        raster = raster.astype(RasterOps.intermediary_dtype)
        if whiteref_raster.shape != raster.shape:
            whiteref_raster = np.broadcast_to(whiteref_raster, raster.shape)
        corrected = np.divide(raster, whiteref_raster)

        if clip is True:
            corrected = np.clip(corrected, a_min=0, a_max=1)

        return corrected

    @staticmethod
    def correction_whiteref_from_mask(raster, mask, clip=False):
        whiteref = raster[mask > 0]
        return RasterOps.correction_whiteref(
            raster, RasterOps.to_pixel(whiteref), clip=clip
        )

    @staticmethod
    def correction_from_raster_with_mask(raster, correction_raster, mask, clip=False):
        whiteref = correction_raster[mask > 0]
        return RasterOps.correction_whiteref(
            raster, RasterOps.to_pixel(whiteref), clip=clip
        )

    @staticmethod
    def mask(raster, mask, masked_val=None):
        """
        Set all values of raster, where mask is False or 0, to masked_val
        """

        raster[mask < 1] = masked_val
        return raster

    @staticmethod
    def enclose_largest_label(raster):
        """
        Encloses largest label area in image with smallest possible circle
        """

    @staticmethod
    def mtvi2(raster, wavelengths) -> np.ndarray:
        p800 = raster[:, :, RasterOps.band_idx(wavelengths, 800)]
        p550 = raster[:, :, RasterOps.band_idx(wavelengths, 550)]
        p670 = raster[:, :, RasterOps.band_idx(wavelengths, 670)]

        return np.divide(
            (1.5 * (1.2 * (p800 - p550) - 2.5 * (p670 - p550))),
            (np.sqrt(np.square(2 * p800 + 1) - (6 * p800 - 5 * np.sqrt(p670)) - 0.5)),
        )

    @staticmethod
    def cube2pca(raster, n_components=30, n_components_keep=6, ignore_mask=None):
        scaler = StandardScaler()
        pca = PCA(n_components)
        pipeline = make_pipeline(pca)
        raster_shape = raster.shape
        if ignore_mask is not None:
            raster = raster[ignore_mask > 0][:]
        else:
            raster = np.reshape(
                raster, (raster.shape[0] * raster.shape[1], raster.shape[-1])
            )
        X = pipeline.fit_transform(raster)[:, : n_components_keep + 1]
        features = range(pca.n_components_)
        # plt.bar(features, pca.explained_variance_)
        # plt.xlabel('PCA feature')
        # plt.ylabel('variance')
        # plt.xticks(features)

        # plt.figure()
        # X = pipeline.inverse_transform(X)
        # err = RasterOps.normalize(np.amax(np.abs(np.reshape(reshaped - X, raster.shape)), axis=-1), min=0, max=2**16-1)
        # plt.imshow(err)
        # plt.colorbar()

        # plt.figure()
        # plt.plot(wavelengths, reshaped[1000, :])
        # plt.plot(wavelengths, X[1000, :])
        # plt.show()

        if ignore_mask is not None:
            raster = (
                np.zeros((raster_shape[0], raster_shape[1], n_components_keep)) * np.nan
            )
            raster[ignore_mask > 0] = X
        else:
            raster = np.reshape(
                X,
                (raster_shape[0], raster_shape[1], n_components_keep),
            )
        return (raster, pca.components_[: n_components_keep + 1], pca.mean_, pca)

    @staticmethod
    def continuum_removal(raster, wavelengths):
        endsr = np.zeros((raster.shape[0], raster.shape[1], 2))
        endsr[:, :, 0] = raster[:, :, 0]
        endsr[:, :, 1] = raster[:, :, -1]

        endsw = [wavelengths[0], wavelengths[-1]]

        # y = mx+b
        m = np.divide(endsr[:, :, 1] - endsr[:, :, 0], endsw[1] - endsw[0])
        b = endsr[:, :, 1] - m * endsw[1]
        b = b[:, :, np.newaxis]
        m = m[:, :, np.newaxis]

        interp = m * wavelengths + b
        cr = np.abs(np.divide(raster, interp) - 1)

        # only use removal where it has resulted in values less than 1
        #   to ensure convex hull constraint
        raster[cr <= 1] = cr[cr <= 1]
        return (raster, wavelengths)

    @staticmethod
    def integrate(raster):
        return np.sum(raster, axis=-1)

    @staticmethod
    def linear_interpolate(raster, x1, x2):
        # for every interpolation point, calcuate a spatial linear function
        #   this is faster than calculating full interpolation pixel-by-pixel
        m_cube = np.zeros((raster.shape[0], raster.shape[1], len(x2)))
        b_cube = np.zeros((raster.shape[0], raster.shape[1], len(x2)))

        raster = raster.astype(np.float32)
        for idx, x in enumerate(x2):
            closest_band = RasterOps.band_idx(x1, x)
            if x1[closest_band] >= x:
                upper_bound = closest_band
                lower_bound = upper_bound if upper_bound == 0 else upper_bound - 1
            else:
                lower_bound = closest_band
                upper_bound = lower_bound if lower_bound == len(x1) else lower_bound + 1

            # y = mx+b
            assert x1[upper_bound] >= x
            assert x1[lower_bound] <= x
            if lower_bound == upper_bound:
                m = np.zeros((raster.shape[0], raster.shape[1]))
                b = np.zeros((raster.shape[0], raster.shape[1]))
            else:
                m = np.divide(
                    raster[:, :, upper_bound] - raster[:, :, lower_bound],
                    x1[upper_bound] - x1[lower_bound],
                )
                b = raster[:, :, upper_bound] - m * x1[upper_bound]
            m_cube[:, :, idx] = m
            b_cube[:, :, idx] = b

        interp = m_cube * x2 + b_cube
        return interp

    @staticmethod
    def get_channels(cube, wavelengths, channels: tuple):
        log.info(f"Selecting bands closest to {channels}")
        log.info(f"Source bands: {wavelengths}")
        bands = [(np.abs(wavelengths - c)).argmin() for c in channels]

        log.info(f"Selected band indices: {bands}")
        raster_new = np.zeros(
            (cube.shape[0], cube.shape[1], len(channels)), dtype=cube.dtype
        )

        for i, c in enumerate(channels):
            raster_new[:, :, i] = cube[:, :, bands[i]]

        return raster_new

    @staticmethod
    def render_with_sensitivities(
        radiance=None, reflectance=None, illuminant_spd=None, sensitivities=None
    ):
        if radiance is None and (reflectance is None or illuminant_spd is None):
            raise UserWarning(
                "Must provide either scene radiance or reflectance and illuminant_spd"
            )

        if radiance is not None:
            energy = radiance
        else:
            energy = reflectance * illuminant_spd

        return np.matmul(energy, sensitivities)


class TiffOptions(NamedTuple):
    compression_mode: int = 0
    rgb_only: bool = False


class STIFF_TYPE(Enum):
    RADIOMETRY = 1
    PCA = 2


class STiff:
    type: STIFF_TYPE = STIFF_TYPE.RADIOMETRY

    _cube: Any = None

    @property
    def cube(self):
        if self._cube is None:
            self.read()
        return self._cube

    @cube.setter
    def cube(self, value):
        self._cube = value

    _extras: Any = None

    @property
    def extras(self):
        if self._extras is None:
            self.read()
        return self._extras

    @extras.setter
    def extras(self, value):
        self._extras = value

    _wavelengths: np.ndarray[float, np.dtype[np.float32]] = None

    @property
    def wavelengths(self):
        if self._wavelengths is None:
            self.read()
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    _rgb: Any = None

    @property
    def rgb(self):
        if self._rgb is None:
            self.read()
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    _metadata: str = None

    @property
    def metadata(self):
        if self._metadata is None:
            self.read()
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    filepath: Path
    tiff_options: TiffOptions
    _masks: Dict = None

    @property
    def masks(self):
        if self._masks is None or len(self._masks.keys()) == 0:
            self.read_masks()
        return self._masks

    @masks.setter
    def masks(self, value):
        self._masks = value

    mtiffpath: Path = None

    def __init__(
        self,
        filepath: str | Path,
        tiff_options: TiffOptions = TiffOptions(),
        mtiffpath: str | Path = None,
    ):
        if mtiffpath is not None:
            self.mtiffpath = Path(mtiffpath)
        self.filepath = Path(filepath)

        self.tiff_options = tiff_options
        self.masks = {}
        self.extras = {}

    def read(self, read_masks: str | Path = None):
        # read stiff file data
        c, e, w, r, m = read_stiff(
            self.filepath.as_posix(), rgb_only=self.tiff_options.rgb_only
        )
        self.cube = c
        self.extras = e
        self.wavelengths = w.astype(np.float32)
        self.rgb = r
        self.metadata = m

        if "components" in self.extras.keys():
            self.type = STIFF_TYPE.PCA

        # read mtiff file data
        if read_masks is None:
            return self

        mtiffpath = self.mtiffpath
        if type(read_masks) != bool:
            mtiffpath = Path(read_masks)

        if mtiffpath is not None:
            self.read_masks(Path(read_masks))

        return self

    def read_masks(self, maskpath: str | Path = None):
        if maskpath is None:
            maskpath = self.mtiffpath
        self.masks = read_mtiff(Path(maskpath).as_posix())
        return self

    def write_stiff(self, extra_tags=None):
        if not self.filepath.parent.exists():
            makedirs(self.filepath.parent.as_posix())

        assert self._cube is not None
        assert self._extras is not None
        assert self._wavelengths is not None
        assert self._rgb is not None

        write_stiff(
            self.filepath.as_posix(),
            self._cube,
            self._extras,
            self._wavelengths,
            self._rgb,
            self._metadata,
            extra_tags=extra_tags,
            compression=self.tiff_options.compression_mode,
        )

    def write_mtiff(self, path: str | Path):
        path = Path(path)

        if not path.parent.exists():
            makedirs(path.parent.as_posix())

        write_mtiff(path.as_posix(), self.masks)

    def reconstruct_from_pca(self):
        c = np.dot(self.cube, self.extras["components"]) + self.extras["mean"]
        return c

    def render8bit(self, channels=(650, 550, 450)):
        if self.type is STIFF_TYPE.RADIOMETRY:
            c = self.cube
            w = self.wavelengths
            rgb = RasterOps.normalize(
                np.nan_to_num(RasterOps.get_channels(c, w, channels), nan=0),
                asinttype=np.uint8,
            )
        elif self.type is STIFF_TYPE.PCA:
            nan_mask = np.invert(np.isnan(self.cube[:, :, 0]))
            c = self.reconstruct_from_pca()

            w = np.squeeze(self.extras["wavelengths"])
            rgb = RasterOps.normalize(
                np.nan_to_num(RasterOps.get_channels(c, w, channels), nan=0),
                asinttype=np.uint8,
            )
            rgb[nan_mask == False] = 0

        self.rgb = rgb

    def copy(self):
        return copy(self)
