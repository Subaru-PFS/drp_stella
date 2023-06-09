import lsst.utils

from .fluxModelSet import FluxModelSet

from deprecated import deprecated
import numpy as np
from scipy.interpolate import RBFInterpolator

import argparse
import json
import os
import pickle
import textwrap


@deprecated(
    reason="It is not necessary to run makeFluxModelInterpolator"
    " with fluxmodeldata >= ambre-20230608."
    " See PIPE2D-1231."
)
def makeFluxModelInterpolator(fluxmodeldataPath):
    """Generate an RBF interpolation model from input model spectra.

    Parameters
    ----------
    fluxmodeldataPath : `str`
        Path to ``fluxmodeldata`` package.
    nProcs : `int`
        Number of processes.
    """
    if os.path.exists(os.path.join(fluxmodeldataPath, "pca.fits")):
        print("This program no longer needs running. See PIPE2D-1231.")
        return

    modelSet = FluxModelSet(fluxmodeldataPath)

    if "isinterpolated" in modelSet.parameters.dtype.names:
        inputParameters = modelSet.parameters[modelSet.parameters["isinterpolated"] == 0]
    else:
        inputParameters = modelSet.parameters

    inputParameters.sort(order=["teff", "logg", "m", "alpha"])

    # make an empty array for input fluxes
    param = inputParameters[0]
    spectrum = modelSet.getSpectrum(param["teff"], param["logg"], param["m"], param["alpha"])
    fluxList = np.empty(shape=(len(inputParameters), len(spectrum)), dtype=float)
    wcs = spectrum.wavelength.toFitsHeader()

    # Store fluxes of each model into a single array
    for i, param in enumerate(inputParameters):
        spectrum = modelSet.getSpectrum(param["teff"], param["logg"], param["m"], param["alpha"])
        fluxList[i, :] = spectrum.flux

    path = os.path.join(fluxmodeldataPath, "interpolator-hyperparams.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            hyperparams = json.load(f)
        kernel = hyperparams["kernel"]
        teffScale = hyperparams["teffScale"]
        loggScale = hyperparams["loggScale"]
        mScale = hyperparams["mScale"]
        alphaScale = hyperparams["alphaScale"]
        epsilon = hyperparams["epsilon"]
        fluxScale = hyperparams["fluxScale"]
    else:
        print('Hyperparameter file not found. (Maybe a "small" fluxmodeldata package is set up)')
        print("Default hyperparameters will be used.")
        # Default hyperparams found by means of cross-validation
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
        pickle.dump(
            {
                "wcs": wcs.tostring(),
                "interpolator": interpolator,
                "kernel": kernel,
                "teffScale": teffScale,
                "loggScale": loggScale,
                "mScale": mScale,
                "alphaScale": alphaScale,
                "epsilon": epsilon,
                "fluxScale": fluxScale,
                "lenWavelength": fluxList.shape[-1],
            },
            f,
            protocol=4,
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Make `interpolator.pickle` in `fluxmodeldata` package.
            This pickle file is required by `fitPfsFluxReference.py`.
        """
        ),
    )
    parser.add_argument("fluxmodeldata", nargs="?", help="Path to fluxmodeldata package")
    args = parser.parse_args()

    if not args.fluxmodeldata:
        args.fluxmodeldata = lsst.utils.getPackageDir("fluxmodeldata")

    makeFluxModelInterpolator(args.fluxmodeldata)
