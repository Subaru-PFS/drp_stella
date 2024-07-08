from typing import ClassVar, List, Type

import numpy as np

import lsstDebug
from lsst.pex.config import Config, DictField, Field, ListField
from lsst.pipe.base import Task, Struct

from lsst.afw.image import MaskedImage
from pfs.datamodel import FiberStatus
from .datamodel import PfsConfig
from pfs.drp.stella import DetectorMap
from pfs.drp.stella.utils.plotting import addColorbar

__all__ = ("BackgroundConfig", "BackgroundTask")


class BackgroundConfig(Config):
    xOrder = Field(dtype=int, default=2, doc="Order of polynomial in x")
    yOrder = Field(dtype=int, default=15, doc="Order of polynomial in y")
    mask = ListField(dtype=str, default=["BAD", "NO_DATA"], doc="Mask planes to ignore")
    radius = Field(dtype=float, default=7, doc="Radius of background region to mask (pixels)")


class BackgroundTask(Task):
    ConfigClass = BackgroundConfig
    _DefaultName = "background"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, image: MaskedImage, detectorMap: DetectorMap) -> Struct:
        """Fit and subtract background to an image

        We mask out all fibers, fit a 2D polynomial to the remaining pixels
        and subtract it from the ``image``.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to fit and subtract background from.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:

            - ``coeff`` (`np.ndarray`): Coefficients of background model.
            - ``model`` (`np.ndarray`): Model image.
            - ``median`` (`float`): Median of background model.
            - ``rms`` (`float`): RMS of residual.
        """
        xNorm, yNorm = np.meshgrid(np.linspace(0, 1, image.getWidth()), np.linspace(0, 1, image.getHeight()))

        badBitmask = image.mask.getPlaneBitMask(self.config.mask)
        select = (image.mask.array & badBitmask) == 0

        for ff in detectorMap.fiberId:
            for row, col in zip(range(image.getHeight()), detectorMap.getXCenter(ff)):
                xStart = int(np.floor(col - self.config.radius))
                xStop = int(np.ceil(col + self.config.radius)) + 1  # exclusive
                select[row, xStart:xStop] = False

        # Construct design matrix
        design = self.makeDesignMatrix(xNorm[select], yNorm[select])
        coeff, r, rank, s = np.linalg.lstsq(design, image.image.array[select], None)
        shape = image.image.array.shape
        model = np.dot(self.makeDesignMatrix(xNorm.flatten(), yNorm.flatten()), coeff).reshape(shape)

        if self.debugInfo.display:
            import matplotlib.cm
            import matplotlib.colors
            import matplotlib.pyplot as plt

            xx, yy = np.meshgrid(np.arange(image.getWidth()), np.arange(image.getHeight()))

            fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
            cmap = matplotlib.cm.rainbow
            norm = matplotlib.colors.Normalize()
            norm.autoscale(image.image.array[select])
            axes[0].scatter(
                xx[select], yy[select], c=image.image.array[select], cmap=cmap, norm=norm, marker="."
            )
            addColorbar(fig, axes[0], cmap, norm, "Flux (electrons)")
            axes[0].set_title("Data")
            axes[1].scatter(xx[select], yy[select], c=model.array[select], cmap=cmap, norm=norm, marker=".")
            addColorbar(fig, axes[1], cmap, norm, "Flux (electrons)")
            axes[1].set_title("Model")
            residual = image.image.array[select] - model[select]
            norm = matplotlib.colors.Normalize()
            norm.autoscale(residual)
            axes[2].scatter(xx[select], yy[select], c=residual, cmap=cmap, norm=norm, marker=".")
            addColorbar(fig, axes[2], cmap, norm, "Flux (electrons)")
            axes[2].set_title("Residual")
            plt.show()

        image.image.array -= model
        median = np.median(model)
        rms = np.std(image.image.array[select])
        self.log.info("Background fit median=%f; residual RMS=%f", median, rms)

        return Struct(coeff=coeff, model=model, median=median, rms=rms)

    def makeDesignMatrix(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """Construct a design matrix for a polynomial fit

        Parameters
        ----------
        xx : `numpy.ndarray`
            X coordinates.
        yy : `numpy.ndarray`
            Y coordinates.

        Returns
        -------
        design : `numpy.ndarray`
            Design matrix.
        """
        design: List[np.ndarray] = []
        for ii in range(self.config.xOrder + 1):
            for jj in range(self.config.yOrder + 1 - ii):
                design.append(xx**ii * yy**jj)
        return np.array(design).T


class DichroicBackgroundConfig(Config):
    """Configuration for background measurement in the dichroic region"""
    fiberHalfWidth = Field(dtype=float, default=15, doc="Radius around fiber to mask (pixels)")
    fiberStatus = ListField(
        dtype=str,
        default=["GOOD", "BROKENFIBER", "BLOCKED"],
        doc="FiberStatus values to use for dichroic background subtraction",
    )
    mask = ListField(
        dtype=str,
        default=["SAT", "BAD", "NO_DATA"],
        doc="Mask planes to ignore for dichroic background subtraction",
    )
    top = DictField(
        keytype=str,
        itemtype=int,
        default=dict(
            b=30,
            r=30,
        ),
        doc="Number of rows to use for dichroic background subtraction at top of detector, indexed by arm",
    )
    bottom = DictField(
        keytype=str,
        itemtype=int,
        default=dict(
            r=50,
            n=50,
        ),
        doc="Number of rows to use for dichroic background subtraction at bottom of detector, indexed by arm",
    )


class DichroicBackgroundTask(Task):
    ConfigClass: ClassVar[Type[Config]] = DichroicBackgroundConfig
    _DefaultName: ClassVar[str] = "background"

    def run(self, image: MaskedImage, arm: str, detectorMap: DetectorMap, pfsConfig: PfsConfig) -> np.ndarray:
        """Remove any diffuse background in the dichroic region

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to process.
        arm : `str`
            Spectrograph arm name (``b``, ``r``, ``n``, ``m``).
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId to x,y on the detector.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        """
        height = image.getHeight()
        topRows = self.config.top.get(arm, None)
        bottomRows = self.config.bottom.get(arm, None)
        if topRows is None and bottomRows is None:
            return np.zeros(height)

        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        fiberId = pfsConfig.select(fiberId=detectorMap.fiberId, fiberStatus=fiberStatus).fiberId
        bitmask = image.mask.getPlaneBitMask(self.config.mask)

        def measureBackground(rows: slice) -> np.ndarray:
            """Measure the background in the given rows

            Parameters
            ----------
            rows : `slice`
                Rows to measure.

            Returns
            -------
            background : `np.ndarray`
                Background as a function of row.
            """
            array = image.image.array[rows, :]
            mask = image.mask.array[rows, :]
            good = (mask & bitmask) == 0

            for ff in fiberId:
                for yy, xCenter in enumerate(detectorMap.getXCenter(ff)[rows]):
                    xLow = int(xCenter - self.config.fiberHalfWidth)
                    xHigh = int(xCenter + self.config.fiberHalfWidth + 0.5)
                    good[yy, xLow:xHigh] = False
            return np.median(array[good])

        top = measureBackground(slice(-topRows, height)) if topRows else None
        bottom = measureBackground(slice(bottomRows)) if bottomRows else None

        if top is not None and bottom is not None:
            # Subtract a ramp across the image
            yBottom = 0.5*bottomRows
            yTop = height - 0.5*topRows
            yy = np.arange(height)
            slope = (top - bottom)/(yTop - yBottom)
            background = (yy - yBottom)*slope + bottom
            self.log.info("Subtracting dichroic: bottom=%f top=%f", bottom, top)
        elif top is not None:
            background = np.full(height, top)
            self.log.info("Subtracting dichroic: top=%f", top)
        elif bottom is not None:
            background = np.full(height, bottom)
            self.log.info("Subtracting dichroic: bottom=%f", bottom)
        else:
            raise AssertionError("Should never get here")

        image.image.array -= background[:, np.newaxis]

        return background
