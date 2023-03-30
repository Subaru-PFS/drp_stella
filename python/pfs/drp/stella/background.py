from typing import List

import numpy as np

import lsstDebug
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task, Struct

from lsst.afw.image import MaskedImage
from pfs.drp.stella import DetectorMap
from pfs.drp.stella.fitDistortedDetectorMap import addColorbar

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
