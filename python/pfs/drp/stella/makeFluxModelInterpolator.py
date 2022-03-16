import lsst.utils

from .fluxModelSet import FluxModelSet
from .utils import parallel

import numpy as np
from scipy.interpolate import Rbf

import argparse
import os
import pickle
import textwrap


def makeFluxModelInterpolator(fluxmodeldataPath, nProcs=1):
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
    inputFlux = np.empty((len(spectrum), len(inputParameters)), dtype=float)
    wcs = spectrum.wavelength.toFitsHeader()

    # Store fluxes of each model into a single array
    for i, param in enumerate(inputParameters):
        spectrum = modelSet.getSpectrum(param["teff"], param["logg"], param["m"], param["alpha"])
        inputFlux[:, i] = spectrum.flux

    teff = np.asarray(inputParameters["teff"], dtype=float) / 1e3
    logg = np.asarray(inputParameters["logg"], dtype=float)
    m = np.asarray(inputParameters["m"], dtype=float)
    alpha = np.asarray(inputParameters["alpha"], dtype=float)

    def makeRBFModel(flux):
        """Make an RBF model at a specific wavelength.

        Parameters
        ----------
        flux : `numpy.array` of `float`
            Fluxes at the various parameters and at a specific wavelength.

        Returns
        -------
        rbf : `scipy.interpolate import Rbf`
            RBF model in charge of a specific wavelength.
        """
        return Rbf(
            teff,
            logg,
            m,
            alpha,
            flux,
            function="multiquadric",
            epsilon=2.0,
        )

    # Execute `makeRBFModel` in parallel
    interpolator = parallel.parallel_map(makeRBFModel, inputFlux, nProcs)

    # Pickle the model
    output = os.path.join(fluxmodeldataPath, "interpolator.pickle")
    with open(output, "wb") as f:
        pickle.dump({
            "wcs": wcs.tostring(),
            "interpolator": interpolator,
        }, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Make `interpolator.pickle` in `fluxmodeldata` package.
            This pickle file is required by `fitPfsFluxReference.py`.

            This program will take 4-5 hours with `--nprocs=128`, as a rough estimate.

            You should disable thread-level parallelism (used by some numpy)
            and use `--nprocs=N` option. We recommend that you invoke this program
            with a command like (bash):

                env OMP_NUM_THREADS=1 %(prog)s --nprocs=$(nproc)
        """)
    )
    parser.add_argument("fluxmodeldata", nargs="?", help="Path to fluxmodeldata package")
    parser.add_argument("-j", "--nprocs", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    if not args.fluxmodeldata:
        args.fluxmodeldata = lsst.utils.getPackageDir("fluxmodeldata")

    makeFluxModelInterpolator(args.fluxmodeldata, args.nprocs)
