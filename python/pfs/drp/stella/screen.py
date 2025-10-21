import os

import numpy as np
from lsst.daf.base import PropertyList
from lsst.obs.pfs.utils import getLamps
from lsst.pex.config import Config
from lsst.pipe.base import Task
from pfs.datamodel import PfsConfig, TargetType, PfsFiberArraySet
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay

__all__ = ("ScreenResponseConfig", "ScreenResponseTask", "ScreenModel")


def bilinearInterpolation(xgrid, ygrid, fields, xy, nearestFallback=True):
    """
    Bilinear eval on a regular grid for one or many fields.

    Parameters
    ----------
    xgrid, ygrid : ndarray
        Grid centers (len nx, ny).
    fields : ndarray
        Shape (nField, ny, nx) or (nField, ny*nx) or (ny, nx) or (ny*nx,).
        Last axis must match nx*ny when flattened.
    xy : ndarray, shape (N, 2)
        Query [[x, y], ...].
    nearestFallback : bool, optional
        If True, clamp out-of-bounds with nearest; else return NaN outside.

    Returns
    -------
    out : ndarray, shape (nField, N)  or (N,) if a single field
    """
    xgrid = np.asarray(xgrid)
    ygrid = np.asarray(ygrid)
    xy = np.asarray(xy)

    nx = xgrid.size
    ny = ygrid.size

    arr = np.asarray(fields)
    if arr.ndim == 1:
        arr = arr.reshape(1, ny, nx)
    elif arr.ndim == 2:
        print(arr.shape)
        if arr.shape[1] == ny * nx:
            arr = arr.reshape(arr.shape[0], ny, nx)
        elif arr.shape == (ny, nx):
            arr = arr.reshape(1, ny, nx)
        else:
            raise ValueError("fields shape incompatible with (ny, nx)")

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    ix = (xy[:, 0] - xgrid[0]) / dx
    iy = (xy[:, 1] - ygrid[0]) / dy
    coords = np.vstack([iy, ix])

    mode = 'nearest' if nearestFallback else 'constant'
    out = [map_coordinates(f, coords, order=1, mode=mode, cval=np.nan, prefilter=False) for f in arr]
    out = np.asarray(out)

    return out[0] if fields.ndim in (1, 2) and fields.shape == (ny, nx) else out


def delaunayInterpolation(XY, fields, xy, nearestFallback=True):
    """
    Piecewise-linear interpolation on a Delaunay triangulation (grid-agnostic).

    Parameters
    ----------
    XY : ndarray, shape (M, 2)
        Source points [[x, y], ...].
    fields : ndarray
        Shape (nField, M) or (M,).
    xy : ndarray, shape (N, 2)
        Query [[x, y], ...].
    nearestFallback : bool, optional
        If True, fill failed linear evals with nearest neighbor.

    Returns
    -------
    out : ndarray, shape (nField, N) or (N,) if a single field
    """
    XY = np.asarray(XY)
    xy = np.asarray(xy)

    arr = np.asarray(fields)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    tri = Delaunay(XY)
    lin = [LinearNDInterpolator(tri, v, fill_value=np.nan) for v in arr]
    out = [f(xy) for f in lin]
    out = np.asarray(out)

    if nearestFallback and np.any(~np.isfinite(out)):
        near = [NearestNDInterpolator(XY, v) for v in arr]
        for i in range(out.shape[0]):
            bad = ~np.isfinite(out[i])
            if np.any(bad):
                out[i, bad] = near[i](xy[bad])

    return out[0] if fields.ndim == 1 else out


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
            self.log.info("Applying new screen response correction to quartz lamp spectra, INSROT=%f", insrot)
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
        """Correct spectra.norm for the *screen response* (store the pattern in norm).

        Conventions (read this first!)
        -------------------------------
        - Response R(x, λ): multiplicative pattern from the screen (quartz flat), defined so that
            quartz_flux / R  ≈  1
          i.e. R is what imprints non-uniformity on quartz exposures.

        - Correction C(x, λ): the inverse of the response:
            C = 1 / R
          C is what you would *multiply* a quartz frame by to flatten it.

        Why `norm /= correction` ?
        --------------------------
        We want `norm` to *carry* the screen pattern used by downstream ratios. If we set
            norm_applied = norm_base / C = norm_base * R
        then for a quartz exposure Q:
            Q / norm_applied = (Q / (norm_base * R))  ≈  R    (since Q / norm_base ≈ 1 initially)
        which is exactly what we want: quartz divided by the applied norm reveals the response.
        For a uniform twilight T:
            T / norm_applied  ≈  1
        if `norm_base` already encapsulates throughput and we only inject/remove the screen via R or C.

        TL;DR: We *divide by C* (i.e. multiply by R) because `norm` is the container for the response pattern.

        Parameters
        ----------
        spectra : PfsFiberArraySet
            Spectra whose .norm will be updated in place to include the screen response R.
        pfsConfig : PfsConfig
            Fiber configuration (used for fiber selection and PFI coordinates).
        insrot : float
            Instrument rotator angle [deg] for the exposure; used to rotate PFI coordinates.

        Raises
        ------
        RuntimeError
            If fiber IDs mismatch or `insrot` is not finite.
        """
        # Align config to spectra; sanity checks
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        if not np.array_equal(pfsConfig.fiberId, spectra.fiberId):
            raise RuntimeError("FiberId mismatch")
        if not np.isfinite(insrot):
            raise RuntimeError("Rotator angle is not finite")

        # Load the screen model (produces R when evaluated); keep as-is for your local test
        screen = ScreenModel.loadModel('/home/alefur/screenModel')

        # Select only “real” science fibers with known positions
        noPosition = np.isnan(pfsConfig.pfiCenter).all(axis=1)
        select = (pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)) & (~noPosition)
        if not np.any(select):
            return

        # Rotate PFI (x, y) by insrot so the screen model is queried in the proper orientation
        xyRot = rotateCoordinatesAroundCenter(
            pfsConfig.pfiCenter[select].T, x0=0, y0=0, theta=np.deg2rad(insrot)
        )

        # Evaluate RESPONSE R(λ, x, y); shapes: (nSel, nλ)
        response = screen.evaluate(spectra.wavelength[select], xy=xyRot.T)

        # Compute CORRECTION C = 1 / R; we *divide norm by C* to *insert R* into norm (see doc above)
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = 1 / response
            bad = ~np.isfinite(correction) | (correction == 0)
            if np.any(bad):
                # If the screen model is undefined somewhere, fall back to neutral correction = 1
                correction[bad] = 1.0

            # The key step: norm_applied = norm_base / C = norm_base * R
            spectra.norm[select] /= correction


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
        scores = self.scores if wavegrid is None else self.interpolateScores(wavegrid)

        if xy is None:
            mean = self.mean  # (nF,)
            components = self.components  # (nC, nF)
        else:
            extended = np.vstack([self.components, self.mean])
            mean, components = self.interpolateComponents(extended, xy)  # (nF,), (nC, nF)

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

    def interpolateComponents(self, extendedComponent, xy, mode='bilinear', nearestFallback=True):
        """
        Interpolate components (last row = mean) at xy using either bilinear or Delaunay.
        """
        xy = np.asarray(xy)

        if mode == 'bilinear':
            xgrid = np.unique(self.x)
            ygrid = np.unique(self.y)
            nx = xgrid.size
            ny = ygrid.size
            if nx * ny != self.x.size:
                raise ValueError("self.x/self.y are not a regular grid; use mode='tri'.")

            fields = np.asarray(extendedComponent)
            if fields.ndim == 2 and fields.shape[1] == ny * nx:
                fields = fields.reshape(fields.shape[0], ny, nx)

            vals = bilinearInterpolation(xgrid, ygrid, fields, xy, nearestFallback=nearestFallback)

        elif mode == 'tri':
            XY = np.column_stack([self.x, self.y])
            fields = np.asarray(extendedComponent)
            vals = delaunayInterpolation(XY, fields, xy, nearestFallback=nearestFallback)

        else:
            raise ValueError("mode must be 'bilinear' or 'tri'")

        return vals[-1], vals[:-1]

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

        print(f"Please help me, Loaded PCA model from {loadDir}")
        return cls(wavelengths, meanResponse, components, scores, x, y)
