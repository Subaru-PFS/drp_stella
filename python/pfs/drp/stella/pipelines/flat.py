import numpy as np
from lsst.afw.image import makeExposure
from lsst.cp.pipe.cpCombine import CalibCombineConfig, CalibCombineConnections, CalibCombineTask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import Struct
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from pfs.datamodel import CalibIdentity

from ..buildFiberProfiles import BuildFiberProfilesTask

__all__ = ("FlatDitherCombineTask", "FlatCombineTask")


class FlatDitherCombineConnections(CalibCombineConnections, dimensions=("instrument", "detector", "dither")):
    """Connections for FlatDitherCombineTask"""

    outputData = OutputConnection(
        name="ditherFlat",
        doc="Output combined dithers.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "dither"),
    )


class FlatDitherCombineConfig(CalibCombineConfig, pipelineConnections=FlatDitherCombineConnections):
    """Configuration for FlatDitherCombineTask"""

    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    minSNR = Field(
        dtype=float,
        default=50,
        doc="Minimum Signal-to-Noise Ratio for normalized flat pixels",
        check=lambda x: x > 0,
    )

    def setDefaults(self):
        super().setDefaults()
        self.profiles.doBlindFind = True  # Because we've dithered the fiber positions
        self.profiles.profileRadius = 2  # Full fiber density, so can't go out very wide
        self.mask = ["BAD", "SAT", "CR", "INTRP"]


class FlatDitherCombineTask(CalibCombineTask):
    """Combine multiple exposures with the same dither setting"""

    ConfigClass = FlatDitherCombineConfig
    _DefaultName = "flatDitherCombine"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        inputs["inputDims"] = [exp.dataId.full for exp in inputRefs.inputExpHandles]
        outputs = self.run(**inputs)
        butler.put(outputs, outputRefs)

    def run(self, inputExpHandles, inputDims):
        """Combine multiple exposures with the same dither setting.

        We also normalise the combined image by the fiber profiles, so the
        result should be an image distributed around unity, at least for the
        lit portions.

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputData``
                Normalised combined exposure generated from the inputs
                (`lsst.afw.image.Exposure`).

        Raises
        ------
        RuntimeError
            Raised if no input data is found.  Also raised if
            config.exposureScaling == InputList, and a necessary scale
            was not found.
        """
        combined = super().run(inputExpHandles, inputDims).outputData

        arm = set(dim["arm"] for dim in inputDims)
        spectrograph = set(dim["spectrograph"] for dim in inputDims)
        assert len(arm) == 1 and len(spectrograph) == 1
        visitInfo = [handle.get(component="visitInfo") for handle in inputExpHandles]
        identity = CalibIdentity(
            obsDate=min(vi.getDate().toPython().date().isoformat() for vi in visitInfo),
            spectrograph=spectrograph.pop(),
            arm=arm.pop(),
            visit0=min(vi.id for vi in visitInfo),
        )

        # Normalise the combined image by the shape of the traces
        profileData = self.profiles.run(combined, identity=identity)
        if len(profileData.profiles) == 0:
            raise RuntimeError("No profiles found")
        self.log.info("%d fiber profiles found", len(profileData.profiles))
        maskVal = combined.mask.getPlaneBitMask(self.config.mask)
        traces = profileData.profiles.makeFiberTraces(combined.getDimensions(), profileData.centers)
        spectra = traces.extractSpectra(combined.maskedImage, maskVal)
        self.log.info("Extracted %d spectra", len(spectra))

        expect = spectra.makeImage(combined.getBBox(), traces)
        # Occasionally NaNs are present in these images,
        # despite the original coadded image containing zero NaNs
        expect.array[~np.isfinite(expect.array)] = 1.0

        with np.errstate(invalid="ignore"):
            snr = combined.image.array / np.sqrt(combined.variance.array)
            bad = (expect.array <= 0.0) | ((combined.mask.array & maskVal) > 0) | (snr < self.config.minSNR)

        combined.image.array[bad] = 0.0
        combined.variance.array[bad] = 0.0
        expect.array[bad] = 1.0
        combined.mask.array &= ~maskVal  # Remove planes we are masking so they don't leak through
        combined.mask.array[bad] |= combined.mask.getPlaneBitMask("NO_DATA")

        combined.maskedImage /= expect

        return Struct(outputData=combined)


class FlatCombineConnections(CalibCombineConnections, dimensions=("instrument", "detector")):
    """Connections for FlatCombineTask"""

    inputExpHandles = InputConnection(
        name="ditherFlat",
        doc="Input combined dithers.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "dither"),
        multiple=True,
        deferLoad=True,
    )
    outputData = OutputConnection(
        name="fiberFlat",
        doc="Combined flat",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class FlatCombineConfig(CalibCombineConfig, pipelineConnections=FlatCombineConnections):
    """Configuration for FlatCombineTask"""

    pass


class FlatCombineTask(CalibCombineTask):
    """Combine normalised dither exposures"""

    ConfigClass = FlatCombineConfig
    _DefaultName = "fiberFlatCombine"

    def run(self, inputExpHandles, inputDims=None):
        """Combine normalised dither exposures

        No fancy rejection is required here: cosmic rays and other artifacts
        are removed by the `FlatDitherCombineTask`. Here, we simply perform a
        weighted sum of the inputs.

        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputData``
                Normalised combined exposure generated from the inputs
                (`lsst.afw.image.Exposure`).

        Raises
        ------
        RuntimeError
            Raised if no input data is found.  Also raised if
            config.exposureScaling == InputList, and a necessary scale
            was not found.
        """
        if not inputExpHandles:
            raise RuntimeError("No inputs provided")

        sumFlat = None
        sumWeight = None
        for handle in inputExpHandles:
            exp = handle.get()
            image = exp.image
            noData = exp.mask.getPlaneBitMask("NO_DATA")
            bad = (exp.mask.array & noData) != 0
            exp.mask.array[bad] &= ~noData
            weight = exp.variance.clone()
            weight.array[:] = 1.0 / weight.array
            weight.array[bad] = 0.0
            image *= weight

            if sumFlat is None:
                sumFlat = exp.maskedImage
                sumWeight = weight
            else:
                sumFlat += exp.maskedImage
                sumWeight += weight

        if sumFlat is None:
            raise RuntimeError("No valid inputs")
        if np.all(sumWeight.array == 0.0):
            raise RuntimeError("No good pixels")

        # Avoid NANs when dividing
        empty = sumWeight.array == 0
        sumFlat.image.array[empty] = 1.0
        sumFlat.variance.array[empty] = 0.0
        sumWeight.array[empty] = 1.0
        sumFlat.mask.addMaskPlane("BAD_FLAT")
        badFlat = sumFlat.mask.getPlaneBitMask("BAD_FLAT")

        sumFlat /= sumWeight
        sumFlat.mask.array[empty] |= badFlat

        flatExp = makeExposure(sumFlat)
        isBad = ~np.isfinite(flatExp.image.array) | ~np.isfinite(flatExp.variance.array)
        flatExp.image.array[isBad] = np.median(flatExp.image.array[~isBad])
        flatExp.variance.array[isBad] = np.median(flatExp.variance.array[~isBad])

        self.combineHeaders(inputExpHandles, flatExp, calibType="flat")

        return Struct(outputData=flatExp)
