import numpy as np

import lsst.utils.tests

from pfs.datamodel import MaskHelper, PfsConfig, TargetType, FiberStatus
from pfs.drp.stella.datamodel import PfsFiberArraySet
from pfs.drp.stella.focalPlaneFunction import FocalPlaneFunction
from pfs.drp.stella.fitFocalPlane import FitFocalPlaneTask
from pfs.drp.stella.fitFocalPlane import FitOversampledSplineTask
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineTask

from pfs.drp.stella.tests import runTests, classParameters

display = None


@classParameters(Task=(FitFocalPlaneTask, FitOversampledSplineTask, FitBlockedOversampledSplineTask),
                 ditherWavelength=(False, True, True))
class FitFocalPlaneTaskTestCase(lsst.utils.tests.TestCase):
    """Test `FitFocalPlaneTask` and subclasses"""
    def setUp(self):
        """Construct spectra and a pfsConfig to use as inputs"""
        self.length = 1234
        self.numFibers = 5
        self.noise = 3.21
        self.minWavelength = 500
        self.maxWavelength = 1000
        self.rng = np.random.RandomState(12345)
        self.actual = 0.1*(np.arange(self.length, dtype=float) - 0.5*self.length)

        identity = dict(visit=12345, arm="r", spectrograph=3)
        fiberId = np.arange(self.numFibers, dtype=int)
        shape = (self.numFibers, self.length)
        wavelength = np.vstack([np.linspace(self.minWavelength, self.maxWavelength, self.length,
                                            dtype=float)]*self.numFibers)
        if self.ditherWavelength:
            dwl = (self.maxWavelength - self.minWavelength)/self.length
            wavelength -= self.rng.uniform(low=-0.5*dwl, high=0.5*dwl, size=self.numFibers)[:, np.newaxis]
        flux = np.vstack([self.actual]*self.numFibers) + self.rng.standard_normal(shape)*self.noise
        mask = np.zeros(shape, dtype=int)
        sky = np.zeros(shape, dtype=float)
        norm = np.ones(shape, dtype=float)
        covar = np.zeros((self.numFibers, 3, self.length), dtype=float)
        covar[:, 0, :] = self.noise**2
        flags = MaskHelper(**{nn: ii for ii, nn in enumerate(("NO_DATA", "SAT", "BAD_FLAT", "CR"))})
        metadata = {}
        self.spectra = PfsFiberArraySet(identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags,
                                        metadata)

        tract = np.full(self.numFibers, 9812, dtype=int)
        patch = ["5,5"]*self.numFibers
        radec = np.zeros(self.numFibers, dtype=float)
        catId = np.full(self.numFibers, 1, dtype=int)
        objId = np.arange(self.numFibers, dtype=int)
        targetType = np.full(self.numFibers, TargetType.SKY, dtype=int)
        fiberStatus = np.full(self.numFibers, FiberStatus.GOOD, dtype=int)
        flux = [np.zeros(0, dtype=float)]*self.numFibers
        filterNames = [[]]*self.numFibers
        position = self.rng.uniform(size=(self.numFibers, 2))
        self.pfsConfig = PfsConfig(123456789, 54321, 0.0, 0.0, 0.0, "brn", fiberId, tract, patch,
                                   radec, radec, catId, objId, targetType, fiberStatus,
                                   flux, flux, flux, flux, flux, flux, filterNames, position, position, None)

    def fit(self, **kwargs) -> FocalPlaneFunction:
        """Fit a `FocalPlaneFunction`

        Parameters
        ----------
        **kwargs
            Task configuration settings.

        Returns
        -------
        function : `FocalPlaneFunction`
            Spectral function fit to the inputs.
        """
        config = self.Task.ConfigClass()
        for name in kwargs:
            setattr(config, name, kwargs[name])
        task = self.Task(name="fit", config=config)
        return task.run(self.spectra, self.pfsConfig)

    def assertFunction(self, function: FocalPlaneFunction, nSigma: float = 4.0):
        """Assert that the function evaluates as expected

        Parameters
        ----------
        function : `FocalPlaneFunction`
            Spectral function fit to the inputs.
        nSigma : `float`
            Number of standard deviations from the expected values beyond which
            no points are permitted.

        Raises
        ------
        AssertionError
            If the function doesn't evaluate correctly.
        """
        evaluation = function(self.spectra.wavelength, self.pfsConfig)
        good = ~evaluation.masks
        self.assertTrue(np.all(np.isfinite(evaluation.values[good])))
        self.assertTrue(np.all(np.isfinite(evaluation.variances[good])))
        self.assertTrue(np.all(evaluation.variances[good] > 0))
        with np.errstate(divide="ignore"):
            residual = (evaluation.values - self.actual[np.newaxis, :])/np.sqrt(evaluation.variances)

        if False:
            import matplotlib.pyplot as plt
            for ii in range(self.numFibers):
                plt.plot(self.spectra.wavelength[ii], self.spectra.flux[ii], "k-")
                plt.plot(self.spectra.wavelength[ii], evaluation.values[ii], "b-")
                if np.any(evaluation.masks[ii]):
                    plt.plot(self.spectra.wavelength[ii][evaluation.masks[ii]],
                             evaluation.values[ii][evaluation.masks[ii]], "bx")
                select = good[ii] & (np.abs(residual[ii]) > nSigma*self.noise)
                if np.any(select):
                    plt.plot(self.spectra.wavelength[ii][select], self.spectra.flux[ii][select], "ro")
            plt.show()

        self.assertTrue(np.all(np.abs(residual[good]) < nSigma*self.noise))

    def testEvaluation(self):
        """Test that the function can be fit and evaluated"""
        func = self.fit()
        self.assertFunction(func)

    def getBadPixels(self, fracBad=0.01):
        """Get an array indicating pixels to flag as bad

        Parameters
        ----------
        fracBad : `float`, optional
            Fraction of pixels to flag as bad.

        Returns
        -------
        bad : `numpy.ndarray` of `bool`
            Pixels to flag as bad.
        """
        numPixels = self.spectra.flux.size
        bad = np.zeros(numPixels, dtype=bool)
        bad[:int(fracBad*numPixels + 0.5)] = True
        self.rng.shuffle(bad)
        return np.reshape(bad, self.spectra.flux.shape)

    def testMasking(self):
        """Test that masks are respected"""
        fracBad = 0.01
        badValue = 100.0

        bad = self.getBadPixels(fracBad)
        self.spectra.flux[bad] += badValue

        self.spectra.mask[bad] = self.spectra.flags["CR"]
        func = self.fit(rejIterations=0)  # Turning off rejection
        self.assertFunction(func)

    def testRejection(self):
        """Test that we can reject bad points"""
        fracBad = 0.01
        badValue = 100.0

        bad = self.getBadPixels(fracBad)
        self.spectra.flux[bad] += badValue

        func = self.fit()
        self.assertFunction(func)

    def testPersistence(self):
        """Test that persistence (with readFits, writeFits) works"""
        func = self.fit()
        with lsst.utils.tests.getTempFilePath("fits") as path:
            func.writeFits(path)
            copy = FocalPlaneFunction.readFits(path)
        self.assertFunction(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
