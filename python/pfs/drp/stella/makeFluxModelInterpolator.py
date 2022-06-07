import lsst.utils

from .fluxModelSet import FluxModelSet

import numpy as np
from scipy.interpolate import RBFInterpolator

import argparse
import os
import pickle
import textwrap


def makeFluxModelInterpolator(fluxmodeldataPath):
    """Generate an RBF interpolation model from input model spectra.

    Parameters
    ----------
    fluxmodeldataPath : `str`
        Path to ``fluxmodeldata`` package.
    nProcs : `int`
        Number of processes.
    """
    modelSet = FluxModelSet(fluxmodeldataPath)
    inputParameters = modelSet.parameters

    # make an empty array for input fluxes
    param = inputParameters[0]
    spectrum = modelSet.getSpectrum(param["teff"], param["logg"], param["m"], param["alpha"])
    fluxList = np.empty(shape=(len(inputParameters), len(spectrum)), dtype=float)
    wcs = spectrum.wavelength.toFitsHeader()

    # Store fluxes of each model into a single array
    for i, param in enumerate(inputParameters):
        spectrum = modelSet.getSpectrum(param["teff"], param["logg"], param["m"], param["alpha"])
        fluxList[i, :] = spectrum.flux

    # Best hyperparams found by means of cross-validation
    kernel = "multiquadric"
    teffScale = 0.0005
    loggScale = 0.5
    mScale = 2.0
    alphaScale = 0.5
    epsilon = 2.0

    fluxScale = np.nanmedian(fluxList)
    fluxList *= 1.0 / fluxScale

    xList = np.empty(shape=(len(inputParameters), 4), dtype=float)
    xList[:, 0] = teffScale * np.asarray(inputParameters["teff"], dtype=float)
    xList[:, 1] = loggScale * np.asarray(inputParameters["logg"], dtype=float)
    xList[:, 2] = mScale * np.asarray(inputParameters["m"], dtype=float)
    xList[:, 3] = alphaScale * np.asarray(inputParameters["alpha"], dtype=float)

    interpolator = RBFInterpolator(xList, fluxList, kernel=kernel, epsilon=epsilon)

    # Pickle the model
    output = os.path.join(fluxmodeldataPath, "interpolator.pickle")
    with open(output, "wb") as f:
        pickle.dump({
            "wcs": wcs.tostring(),
            "interpolator": interpolator,
            "kernel": kernel,
            "teffScale": teffScale,
            "loggScale": loggScale,
            "mScale": mScale,
            "alphaScale": alphaScale,
            "epsilon": epsilon,
            "fluxScale": fluxScale,
            "lenWavelength": fluxList.shape[-1]
        }, f, protocol=4)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Make `interpolator.pickle` in `fluxmodeldata` package.
            This pickle file is required by `fitPfsFluxReference.py`.
        """)
    )
    parser.add_argument("fluxmodeldata", nargs="?", help="Path to fluxmodeldata package")
    args = parser.parse_args()

    if not args.fluxmodeldata:
        args.fluxmodeldata = lsst.utils.getPackageDir("fluxmodeldata")

    makeFluxModelInterpolator(args.fluxmodeldata)
