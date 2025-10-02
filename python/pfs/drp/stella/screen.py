import os

import numpy as np
from lsst.daf.base import PropertyList
from lsst.obs.pfs.utils import getLamps
from lsst.pex.config import Config
from lsst.pipe.base import Task
from pfs.datamodel import *
from pfs.datamodel import PfsConfig, TargetType, PfsFiberArraySet
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

__all__ = ("ScreenResponseConfig", "ScreenResponseTask", "ScreenModel")


class ScreenResponseConfig(Config):
    pass


class ScreenResponseTask(Task):
    ConfigClass = ScreenResponseConfig
    _DefaultName = "screenResponse"

    def run(self, metadata: PropertyList, spectra: PfsFiberArraySet, pfsConfig: PfsConfig):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata for the exposure.
        spectra : `PfsFiberArraySet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        """
        if not self.isQuartz(metadata):
            self.log.debug("Not applying screen response correction since not a quartz lamp exposure")
            return
        insrot = metadata["INSROT"]
        if np.sum(pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)) > 0:
            self.log.info("Applying screen response correction to quartz lamp spectra, INSROT=%f", insrot)
            self.apply(spectra, pfsConfig, insrot)

    def isQuartz(self, metadata: PropertyList) -> bool:
        """Return whether the exposure is a quartz lamp exposure

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata for the exposure.

        Returns
        -------
        isQuartz : `bool`
            Whether the exposure is a quartz lamp exposure.
        """
        lamps = getLamps(metadata)
        return bool(lamps & set(("Quartz", "Quartz_eng")))

    def apply(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, insrot: float):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        insrot : `float`
            The instrument rotator angle (degrees) of the exposure.
        """
        # Apply screen response correction
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        if not np.array_equal(pfsConfig.fiberId, spectra.fiberId):
            raise RuntimeError("FiberId mismatch")
        if not np.isfinite(insrot):
            raise RuntimeError("Rotator angle is not finite")

        # just testing
        screen = ScreenModel.loadModel('/home/alefur/screenModel')

        # masking irrelevant fibers.
        noPosition = np.isnan(pfsConfig.pfiCenter).all(axis=1)
        select = (pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)) & (~noPosition)

        xyRot = rotateCoordinatesAroundCenter(pfsConfig.pfiCenter[select].T, x0=0, y0=0,
                                              theta=np.deg2rad(insrot))

        correction = screen.evaluate(spectra.wavelength[select], xy=xyRot.T)
        spectra.flux[select] /= correction


def rotationMatrix(theta: float) -> np.ndarray:
    """Compute a 2D rotation matrix for a given angle.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians.

    Returns
    -------
    matrix : `numpy.ndarray`
        A 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotateCoordinatesAroundCenter(x: np.ndarray, x0: float, y0: float, theta: float) -> np.ndarray:
    """Rotate the given coordinates around the specified center.

    Parameters
    ----------
    x : `numpy.ndarray`
        The coordinates to be rotated, in the format
        ``[[x1, x2, ..., xn], [y1, y2, ..., yn]]``.
    x0, y0 : `float`
        The x and y coordinates of the center around which the rotation is
        performed.
    theta : `float`
        The rotation angle in radians.

    Returns
    -------
    xRot : `numpy.ndarray`
        The rotated coordinates in the same format as the input `x`.
    """
    rotation = rotationMatrix(theta)
    center = np.array(([x0], [y0]))
    xCentered = x - center
    xRot = np.matmul(rotation, xCentered)
    xRot += center
    return xRot


class ScreenModel:
    def __init__(self, wavelengths, meanResponse, components, scores, x, y, nComponents=None):
        """
        Initialize the PCAReconstructor.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of shape (nSamples,) containing the wavelengths used during training.
        meanResponse : np.ndarray
            Mean fiber response used during PCA (shape: nFibers).
        components : np.ndarray
            PCA components (shape: nTotalComponents, nFibers).
        scores : np.ndarray
            PCA projection scores (shape: nSamples, nTotalComponents).
        x, y : np.ndarray
            Positions (shape: nFibers,) corresponding to each column in the components.
        nComponents : int or None
            Number of PCA components to retain. If None, use all components.
        """
        self.wavelengths = wavelengths
        self.mean = meanResponse
        self.components = components[:nComponents] if nComponents is not None else components
        self.scores = scores[:, :nComponents] if nComponents is not None else scores
        self.nComponents = self.components.shape[0]

        self.x = x
        self.y = y
        self.nFibers = self.mean.size
        self.nSamples = self.scores.shape[0]

    def evaluate(self, wavegrid=None, xy=None):
        """
        Reconstruct fiber response; returns (nFibers, nWaves). Uses batched matmul with simple reshapes.
        """
        scores = self.scores if wavegrid is None else self.interpolateScores(wavegrid)  # (nW,nC) or (nF,nW,nC)

        if xy is None:
            mean = self.mean  # (nF,)
            components = self.components  # (nC, nF)
        else:
            extended = np.vstack([self.components, self.mean])
            mean, components = self.interpolateComponents(xy, extended)  # (nF,), (nC, nF)

        compPerFiber = components.T[:, :, None]

        scoresB = scores[None, :, :] if scores.ndim == 2 else scores
        # final matmul
        recon = (scoresB @ compPerFiber).squeeze(-1)

        return recon + mean[:, None]

    def interpolateScores(self, wavegrid):
        """
        Interpolate PCA scores across wavelength.

        Parameters
        ----------
        wavegrid : np.ndarray
            1D (nW,) or 2D per-fiber (nF, nW)

        Returns
        -------
        np.ndarray
            If 1D: (nW, nC)
            If 2D: (nF, nW, nC)
        """
        wavegrid = np.asarray(wavegrid)
        interpolated_scores = []

        for i in range(self.nComponents):
            f = interp1d(self.wavelengths, self.scores[:, i], kind='cubic',
                         fill_value=(self.scores[0, i], self.scores[-1, i]), bounds_error=False)
            interpolated_scores.append(f(wavegrid))

        if wavegrid.ndim == 1:
            return np.column_stack(interpolated_scores)
        else:
            return np.stack(interpolated_scores, axis=-1)

    def interpolateComponents(self, xy, extendedComponent, method='linear'):
        """
        Interpolate PCA components and mean response at new positions.

        Parameters
        ----------
        xy : np.ndarray
            New (x,y) positions (nFibers,2)
        extendedComponent : np.ndarray
            Stack (nC+1, nFibersOld), last row = mean

        Returns
        -------
        meanInterp : np.ndarray
            Interpolated mean response (nFibers,)
        componentsInterp : np.ndarray
            Interpolated components (nC, nFibers)
        """
        points = np.vstack((self.x, self.y)).T
        interpolated = []

        for values in extendedComponent:
            m = np.isfinite(values)
            evaluated = griddata(points[m], values[m], xy, method=method)

            # fallback for NaN points
            if np.any(~np.isfinite(evaluated)):
                evaluated[~np.isfinite(evaluated)] = griddata(points[m], values[m],
                                                              xy[~np.isfinite(evaluated)], method='nearest')
            interpolated.append(evaluated)

        interpolated = np.array(interpolated)  # (nC+1, nFibers)
        return interpolated[-1], interpolated[:-1]

    def saveModel(self, exportDir):
        """Save PCA model to disk in a new versioned folder."""
        os.makedirs(exportDir, exist_ok=True)
        saveDir = exportDir

        np.save(os.path.join(saveDir, "wavelengths.npy"), self.wavelengths)
        np.save(os.path.join(saveDir, "meanResponse.npy"), self.mean)
        np.save(os.path.join(saveDir, "components.npy"), self.components)
        np.save(os.path.join(saveDir, "scores.npy"), self.scores)
        np.save(os.path.join(saveDir, "x.npy"), self.x)
        np.save(os.path.join(saveDir, "y.npy"), self.y)

        print(f"PCA model saved to {saveDir}")

    @classmethod
    def loadModel(cls, exportDir):
        """Load PCA model from disk."""
        loadDir = exportDir

        wavelengths = np.load(os.path.join(loadDir, "wavelengths.npy"))
        meanResponse = np.load(os.path.join(loadDir, "meanResponse.npy"))
        components = np.load(os.path.join(loadDir, "components.npy"))
        scores = np.load(os.path.join(loadDir, "scores.npy"))
        x = np.load(os.path.join(loadDir, "x.npy"))
        y = np.load(os.path.join(loadDir, "y.npy"))

        print(f"Loaded PCA model from {loadDir}")
        return cls(wavelengths, meanResponse, components, scores, x, y)
