import lsst.utils.tests

from pfs.drp.stella.fitBroadbandSED import FitBroadbandSEDConfig, FitBroadbandSEDTask
from pfs.drp.stella.tests import runTests
from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType

import numpy


class FitBroadbandSEDTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.fitBroadbandSED = FitBroadbandSEDTask(config=FitBroadbandSEDConfig())

    def testRun(self):
        """Test run() method
        """
        random = numpy.random.RandomState(0xfeed5eed)
        filters = ["g", "r", "i", "z", "y"]
        filters = [self.fitBroadbandSED.config.filterMappings.get(f, f) for f in filters]

        trueSEDIndices = numpy.arange(len(self.fitBroadbandSED.fluxLibrary))[::5]
        nFibers = len(trueSEDIndices)

        filterNames = []
        fiberFlux = []
        fiberFluxErr = []
        for record in self.fitBroadbandSED.fluxLibrary[trueSEDIndices]:
            numFilters = random.randint(len(filters))
            amplitude = random.uniform(0.1, 10)

            random.shuffle(filters)
            selectedFilters = filters[:numFilters]
            selectedFluxes = amplitude * numpy.array([record[f] for f in selectedFilters], dtype=float)
            selectedFluxErrs = numpy.abs(selectedFluxes) * random.uniform(0.01, 0.1, size=(numFilters,))

            filterNames.append(selectedFilters)
            fiberFlux.append(selectedFluxes)
            fiberFluxErr.append(selectedFluxErrs)

        allPatches = [
            f"{x},{y}" for x, y in numpy.broadcast(numpy.arange(32).reshape(-1, 1),
                                                   numpy.arange(32).reshape(1, -1))
        ]
        allPatches = numpy.array(allPatches, dtype=(str, 8))

        allTargetTypes = numpy.array([int(x) for x in TargetType], dtype=numpy.int32)
        allFiberStatuses = numpy.array([int(x) for x in FiberStatus], dtype=numpy.int32)

        pfsConfig = PfsConfig(
            pfsDesignId=1,
            visit0=1,
            raBoresight=1.0,
            decBoresight=1.0,
            posAng=1.0,
            arms="brn",
            fiberId=numpy.arange(nFibers, dtype=numpy.int32),
            tract=random.randint(20000, size=(nFibers,), dtype=numpy.int32),
            patch=random.choice(allPatches, size=(nFibers,)),
            ra=random.uniform(360, size=(nFibers,)),
            dec=random.uniform(-90, 90, size=(nFibers,)),
            catId=random.randint(0x8000_0000, size=(nFibers,), dtype=numpy.int32),
            objId=random.randint(0x8000_0000_0000_0000, size=(nFibers,), dtype=numpy.int64),
            targetType=random.choice(allTargetTypes, size=(nFibers,)),
            fiberStatus=random.choice(allFiberStatuses, size=(nFibers,)),
            fiberFlux=fiberFlux,
            psfFlux=fiberFlux,
            totalFlux=fiberFlux,
            fiberFluxErr=fiberFluxErr,
            psfFluxErr=fiberFluxErr,
            totalFluxErr=fiberFluxErr,
            filterNames=filterNames,
            pfiCenter=random.uniform(1000, size=(nFibers, 2)),
            pfiNominal=random.uniform(1000, size=(nFibers, 2)),
            guideStars=None,
        )

        pdfs = self.fitBroadbandSED.run(pfsConfig)

        self.assertEqual(len(pdfs), nFibers)

        for trueSED, targetType, fiberFlux, pdf \
                in zip(trueSEDIndices, pfsConfig.targetType, pfsConfig.fiberFlux, pdfs):
            if TargetType(targetType) == TargetType.FLUXSTD:
                if len(fiberFlux) >= 3:
                    # If three or more bands are available,
                    # the true SED must be preferred the best.
                    self.assertEqual(trueSED, numpy.argmax(pdf))
                elif len(fiberFlux) >= 2:
                    # If only two bands are available,
                    # there may be two or more SEDs
                    # whose probabilities tie for first place.
                    # Still, the true SED must be one of them.
                    self.assertFloatsAlmostEqual(pdf[trueSED] / numpy.max(pdf), 1)
                else:
                    # If no flux is available, or only one flux is available,
                    # FitBroadbandSEDTask must not prefer any SED to the others.
                    self.assertFloatsAlmostEqual(pdf, pdf[0])
            else:
                self.assertTrue(pdf is None)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
