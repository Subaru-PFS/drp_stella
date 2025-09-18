from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping

import numpy as np
from collections import defaultdict, Counter

from lsst.pex.config import ConfigurableField, ListField, Field
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from lsst.geom import SpherePoint, averageSpherePoint, degrees

from pfs.datamodel import Target, Observations, PfsConfig, Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import TargetType, FiberStatus
from pfs.datamodel.drp import PfsCoadd
from pfs.drp.stella.datamodel.drp import PfsArm

from .datamodel import PfsObject, PfsSingle
from .fitFluxCal import calibratePfsArm
from .wavelengthSampling import WavelengthSamplingTask
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions
from .utils.math import fitScales
from .lsf import Lsf, LsfDict, CoaddLsf
from .gen3 import DatasetRefList, zipDatasetRefs
from .fitContinuum import FitContinuumTask

if TYPE_CHECKING:
    from pfs.datamodel import PfsFiberArray

__all__ = ("CoaddSpectraConfig", "CoaddSpectraTask")


class SetWithNaN:
    """A subset of set which includes at most one NaN, despite the fact that NaN != NaN"""

    def __init__(self, iterable):
        """Construct a set from iterable with at most one NaN"""
        self.__set = set()
        sawNaN = False

        for x in iterable:
            if np.isnan(x):
                if not sawNaN:
                    self.__set.add(x)
                    sawNaN = True
            else:
                self.add(x)

    def add(self, val):
        if np.isnan(val) and np.isnan(list(self)).any():
            return

        self.__set.add(val)

    def __len__(self):
        return self.__set.__len__()

    def __repr__(self):
        return self.__set.__repr__()

    def pop(self):
        return self.__set.pop()


class CoaddSpectraConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "combination", "cat_id", "obj_group"),
):
    """Connections for CoaddSpectraTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    objectGroupMap = PrerequisiteConnection(
        name="objectGroupMap",
        doc="Object group map",
        storageClass="ObjectGroupMap",
        dimensions=("instrument", "combination", "cat_id"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    pfsArmLsf = InputConnection(
        name="pfsArmLsf",
        doc="1d line-spread function for extracted spectra",
        storageClass="LsfDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    sky1d = InputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    fluxCal = InputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit"),
        multiple=True,
    )

    pfsCoadd = OutputConnection(
        name="pfsCoadd",
        doc="Flux-calibrated coadded object spectra",
        storageClass="PfsCoadd",
        dimensions=("instrument", "combination", "cat_id", "obj_group"),
    )
    pfsCoaddLsf = OutputConnection(
        name="pfsCoaddLsf",
        doc="Line-spread function for pfsCoadd",
        storageClass="LsfDict",
        dimensions=("instrument", "combination", "cat_id", "obj_group"),
    )


class CoaddSpectraConfig(PipelineTaskConfig, pipelineConnections=CoaddSpectraConnections):
    """Configuration for CoaddSpectraTask"""
    wavelength = ConfigurableField(target=WavelengthSamplingTask, doc="Wavelength sampling")
    mask = ListField(dtype=str, default=["NO_DATA", "SUSPECT", "BAD_SKY", "BAD_FLUXCAL", "BAD_FIBERNORMS"],
                     doc="Mask values to reject when combining")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doSmoothWeights = Field(dtype=bool, default=True, doc="Smooth weights before combining?")
    smoothWeights = ConfigurableField(target=FitContinuumTask, doc="Smoothing of weights")
    applyFluxCalError = Field(dtype=bool, default=False, doc="Propagate flux calibration errors?")
    normIterations = Field(
        dtype=int, default=5, doc="Maximum number of iterations for normalization measurement"
    )
    normSigNoiseThreshold = Field(
        dtype=float, default=3.0, doc="Pixel signal-to-noise threshold for normalization measurement"
    )
    normThreshold = Field(dtype=float, default=2.5, doc="Normalization significance threshold for acceptance")
    normConvergence = Field(
        dtype=float, default=1e-3, doc="Convergence threshold for normalization measurement"
    )


class CoaddSpectraTask(PipelineTask):
    """Coadd multiple observations"""
    _DefaultName = "coaddSpectra"
    ConfigClass = CoaddSpectraConfig

    config: CoaddSpectraConfig
    wavelength: WavelengthSamplingTask
    fluxTable: FluxTableTask
    smoothWeights: FitContinuumTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("wavelength")
        self.makeSubtask("fluxTable")
        self.makeSubtask("smoothWeights")

    def run(self, data: Mapping[Identity, Struct]) -> Struct:
        """Coadd multiple observations

        data : `dict` mapping `Identity` to `Struct`
            Input data. Each element contains data for a single spectrograph
            arm, with attributes:
            - ``identity`` (`Identity`): identity of the data.
            - ``pfsArm`` (`PfsArm`): extracted spectra from spectrograph arm.
            - ``pfsArmLsf`` (`LsfDict`): line-spread function for ``pfsArm``.
            - ``sky1d`` (`FocalPlaneFunction`): 1d sky subtraction model.
            - ``fluxCal`` (`FocalPlaneFunction`): flux calibration solution.
            - ``pfsConfig`` (`PfsConfig`): PFS fiber configuration.

        Returns
        -------
        pfsCoadd : `dict` mapping `Target` to `PfsObject`
            Coadded spectra, indexed by target.
        pfsCoaddLsf : `LsfDict`
            Line-spread functions for coadded spectra, indexed by target.
        """
        targetTypes = (TargetType.SCIENCE, TargetType.FLUXSTD, TargetType.SKY)
        targetSources = defaultdict(list)
        for identity, dd in data.items():
            for target in dd.pfsConfig.select(fiberStatus=FiberStatus.GOOD, targetType=targetTypes):
                targetSources[target].append(identity)

        pfsCoadd: Dict[Target, PfsObject] = {}
        pfsCoaddLsf: Dict[Target, Lsf] = {}
        for target, sources in targetSources.items():
            # if target.objId != 25769807408:
            #     continue
            result = self.process(target, {identity: data[identity] for identity in sources})
            pfsCoadd[target] = result.pfsObject
            pfsCoaddLsf[target] = result.pfsObjectLsf

        return Struct(pfsCoadd=pfsCoadd, pfsCoaddLsf=pfsCoaddLsf)

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        assert butler.quantum.dataId is not None
        ogm = butler.get(inputRefs.objectGroupMap)
        catId = butler.quantum.dataId["cat_id"]
        objGroup = butler.quantum.dataId["obj_group"]
        objId = ogm.objId[ogm.objGroup == objGroup]

        data: Dict[Identity, Struct] = {}
        for pfsConfigRef, pfsArmRef, pfsArmLsfRef, sky1dRef, fluxCalRef in zipDatasetRefs(
            DatasetRefList.fromList(inputRefs.pfsConfig),
            DatasetRefList.fromList(inputRefs.pfsArm),
            DatasetRefList.fromList(inputRefs.pfsArmLsf),
            DatasetRefList.fromList(inputRefs.sky1d),
            DatasetRefList.fromList(inputRefs.fluxCal),
        ):
            pfsConfig: PfsConfig = butler.get(pfsConfigRef)
            pfsArm: PfsArm = butler.get(pfsArmRef).select(pfsConfig, catId=catId, objId=objId)
            identity = pfsArm.identity
            data[identity] = Struct(
                identity=identity,
                pfsArm=pfsArm,
                pfsArmLsf=butler.get(pfsArmLsfRef),
                sky1d=butler.get(sky1dRef),
                fluxCal=butler.get(fluxCalRef),
                pfsConfig=pfsConfig.select(fiberId=pfsArm.fiberId),
            )

        outputs = self.run(data)

        butler.put(PfsCoadd(outputs.pfsCoadd.values(), getPfsVersions()), outputRefs.pfsCoadd)
        butler.put(LsfDict(outputs.pfsCoaddLsf), outputRefs.pfsCoaddLsf)

    def getTarget(self, target: Target, pfsConfigList: List[PfsConfig]) -> Target:
        """Generate a fully-populated `Target` for this target

        We combine the various declarations about the target in the
        ``PfsConfig``s, ensuring the output `Target` has everything it needs
        (e.g., ``targetType``, ``fiberFlux``).

        Parameters
        ----------
        target : `Target`
            Basic identity of target (including ``catId`` and ``objId``).
        pfsConfigList : iterable of `PfsConfig`
            List of top-end configurations. This should include only the target
            of interest.

        Returns
        -------
        result : `Target`
            Fully-populated ``Target``.
        """
        if any(len(cfg) != 1 for cfg in pfsConfigList):
            raise RuntimeError("Multiple fibers included in pfsConfig")
        radec = averageSpherePoint([SpherePoint(cfg.ra[0]*degrees, cfg.dec[0]*degrees) for
                                    cfg in pfsConfigList])

        targetType = Counter([cfg.targetType[0] for cfg in pfsConfigList])
        if len(targetType) > 1:
            self.log.warn("Multiple targetType for target %s (%s); using most common" % (target, targetType))
        targetType = targetType.most_common(1)[0][0]

        fiberFlux = defaultdict(list)
        for pfsConfig in pfsConfigList:
            for ff, flux in zip(pfsConfig.filterNames[0], pfsConfig.fiberFlux[0]):
                fiberFlux[ff].append(flux)
        for ff in fiberFlux:
            flux = SetWithNaN(fiberFlux[ff])
            if len(flux) > 1:
                self.log.warn("Multiple %s flux for target %s (%s); using average" % (ff, target, flux))
                flux = np.average(np.array(fiberFlux[ff]))
            else:
                flux = flux.pop()
            fiberFlux[ff] = flux

        return Target(target.catId, target.tract, target.patch, target.objId,
                      radec.getRa().asDegrees(), radec.getDec().asDegrees(),
                      targetType, dict(**fiberFlux))

    def getObservations(self, dataIdList: Iterable[Identity], pfsConfigList: Iterable[PfsConfig]
                        ) -> Observations:
        """Construct a list of observations of the target

        Parameters
        ----------
        dataIdList : iterable of `Identity`
            List of structs that identify the observation, containing ``visit``,
            ``arm`` and ``spectrograph``.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            List of top-end configurations. This should include only the target
            of interest.

        Returns
        -------
        observations : `Observations`
            Observations of the target.
        """
        if any(len(cfg) != 1 for cfg in pfsConfigList):
            raise RuntimeError("Multiple fibers included in pfsConfig")
        visit = np.array([dataId.visit for dataId in dataIdList])
        arm = [dataId.arm for dataId in dataIdList]
        spectrograph = np.array([dataId.spectrograph for dataId in dataIdList])
        pfsDesignId = np.array([pfsConfig.pfsDesignId for pfsConfig in pfsConfigList])
        fiberId = np.array([pfsConfig.fiberId[0] for pfsConfig in pfsConfigList])
        pfiNominal = np.array([pfsConfig.pfiNominal[0] for pfsConfig in pfsConfigList])
        pfiCenter = np.array([pfsConfig.pfiCenter[0] for pfsConfig in pfsConfigList])
        obsTime = [dataId.obsTime for dataId in dataIdList]
        expTime = np.array([dataId.expTime for dataId in dataIdList])
        return Observations(
            visit, arm, spectrograph, pfsDesignId, fiberId, pfiNominal, pfiCenter, obsTime, expTime
        )

    def getSpectrum(self, target: Target, data: Struct) -> PfsSingle:
        """Return a calibrated spectrum for the nominated target

        Parameters
        ----------
        target : `Target`
            Target of interest.
        data : `Struct`
            Data for a pfsArm containing the object of interest.

        Returns
        -------
        spectrum : `PfsSingle`
            Calibrated spectrum of the target.
        """
        spectrum = data.pfsArm.select(data.pfsConfig, catId=target.catId, objId=target.objId)
        spectrum = calibratePfsArm(
            spectrum,
            data.pfsConfig,
            data.sky1d,
            data.fluxCal,
            applyFluxCalError=self.config.applyFluxCalError,
        )
        return spectrum.extractFiber(PfsSingle, data.pfsConfig, spectrum.fiberId[0])

    def process(self, target: Target, data: Dict[Identity, Struct]) -> Struct:
        """Generate coadded spectra for a single target

        Parameters
        ----------
        target : `Target`
            Target for which to generate coadded spectra.
        data : `dict` mapping `Identity` to `Struct`
            Data from which to generate coadded spectra. These are the results
            from the ``readData`` method.

        Returns
        -------
        pfsObject : `PfsObject`
            Coadded spectrum.
        pfsObjectLsf : `Lsf`
            Line-spread function for coadded spectrum.
        """
        pfsConfigList = [dd.pfsConfig.select(catId=target.catId, objId=target.objId) for dd in data.values()]
        target = self.getTarget(target, pfsConfigList)
        observations = self.getObservations(data.keys(), pfsConfigList)
        wavelength = self.wavelength.run(any(ident.arm == "m" for ident in data))

        spectra = [self.getSpectrum(target, dd) for dd in data.values()]
        lsfList = [dd.pfsArmLsf for dd in data.values()]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectra])
        combination = self.combine(spectra, lsfList, flags, wavelength)
        fluxTable = self.fluxTable.run([dd.getDict() for dd in data.keys()], spectra)

        coadd = PfsObject(target, observations, combination.wavelength, combination.flux,
                          combination.mask, combination.sky, combination.covar, combination.covar2, flags,
                          getPfsVersions(), fluxTable)
        return Struct(pfsObject=coadd, pfsObjectLsf=combination.lsf)

    def combine(self, spectraList, lsfList, flags, wavelength: np.ndarray):
        """Combine spectra

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArray`
            List of spectra to combine for each visit.
        lsfList : iterable of LSF (type TBD)
            List of line-spread functions for each arm for each visit.
        flags : `pfs.datamodel.MaskHelper`
            Mask interpreter, for identifying bad pixels.
        wavelength : `np.ndarray`
            Wavelength array for the combined spectrum.

        Returns
        -------
        flux : `numpy.ndarray` of `float`
            Flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        covar : `numpy.ndarray` of `float`
            Covariance matrix for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        """
        # First, resample to a common wavelength sampling
        resampled = []
        resampledLsf = []
        resampledRange = []
        resampledWeights = []
        for spectrum, lsf in zip(spectraList, lsfList):
            fiberId = spectrum.observations.fiberId[0]
            resampledSpectrum = spectrum.resample(wavelength)
            resampled.append(resampledSpectrum)
            resampledLsf.append(lsf[fiberId].warp(spectrum.wavelength, wavelength))
            minIndex = np.searchsorted(wavelength, spectrum.wavelength[0])
            maxIndex = np.searchsorted(wavelength, spectrum.wavelength[-1])
            resampledRange.append((minIndex, maxIndex))

            with np.errstate(divide="ignore", invalid="ignore"):
                weight = 1.0/resampledSpectrum.covar[0]
            bad = (resampledSpectrum.mask & resampledSpectrum.flags.get(*self.config.mask)) != 0
            if self.config.doSmoothWeights:
                weight = self.smoothWeights.fitArray(weight, bad)
            weight[bad] = 0.0
            resampledWeights.append(weight)

        # Prepare the inputs
        numSpectra = len(spectraList)
        visitMap = defaultdict(list)
        values = defaultdict(list)
        variances = defaultdict(list)
        for ii, ss in enumerate(resampled):
            assert len(spectraList[ii].observations.visit) == 1
            visit = spectraList[ii].observations.visit[0]
            visitMap[visit].append(ii)
            mask = (ss.mask & ss.flags.get(*self.config.mask)) != 0
#            with np.errstate(invalid="ignore", divide="ignore"):
#                mask |= ss.flux**2/ss.variance < self.config.normSigNoiseThreshold**2
            values[visit].append(np.ma.masked_where(mask, ss.flux).reshape(1, -1))
            variances[visit].append(np.ma.masked_where(mask, ss.variance).reshape(1, -1))

        # Iteratively combine and fit for normalization of each visit
        norm = np.ones(numSpectra, dtype=float)
        combination = self.combineResampled(resampled, resampledWeights, 1/norm)
        for ii in range(self.config.normIterations):
            newNorm = norm.copy()
            newErr = np.zeros(numSpectra, dtype=float)
            for visit in visitMap:
                ratio = np.ma.concatenate(values[visit])/combination.flux
                val = np.ma.median(ratio)
                with np.errstate(invalid="ignore", divide="ignore"):
                    for rr, vv, ee in zip(ratio, values[visit], variances[visit]):
                        mm = np.abs(rr - val) > 3.0*np.sqrt(ee)/combination.flux
                        vv.mask |= mm
                        ee.mask |= mm

                val, var = fitScales(
                    values[visit], [combination.flux.reshape(1, -1)]*len(values[visit]), variances[visit]
                )
                residual = ratio - val
                from .utils.math import robustRms
                rms = robustRms(residual.compressed())

                assert len(val) == 1 and len(var) == 1
                nn = val[0]
                ee = np.sqrt(var[0]) + rms

                if nn/ee < self.config.normThreshold:
                    nn = 1.0
                for index in visitMap[visit]:
                    newNorm[index] = nn
                    newErr[index] = ee

            # if np.all(newNorm > 1) or np.all(newNorm < 1):
            #     breakpoint()
            # if np.any(newNorm <= 0):
            #     breakpoint()
            # if np.any(newNorm < 0.1) or np.any(newNorm > 10):
            #     breakpoint()

            change = np.sum(np.abs(newNorm - norm))
            self.log.trace("Iteration %d: norm = %s +/- %s, change = %s", ii, newNorm, newErr, change)
            if change < self.config.normConvergence:
                break
            norm = newNorm
            combination = self.combineResampled(resampled, resampledWeights, 1/norm)
        self.log.trace("Final norms: %s", norm)
        self.log.debug(
            "Normalizations for objId=%s: min, 25%%, 50%%, 75%%, max = %s",
            spectraList[0].target.objId,
            np.percentile(norm, [0, 25, 50, 75, 100]),
        )

        # Calculate the remaining ingredients of the output spectrum
        archetype = spectraList[0]
        combination.mask[~combination.good] |= flags.get("NO_DATA")
        covar = np.zeros((3, combination.variance.size), dtype=archetype.covar.dtype)
        covar[0][combination.good] = combination.variance[combination.good]
        covar[0][~combination.good] = np.inf
        covar[1:2] = np.where(combination.good, 0.0, np.inf)
        covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
        lsf = CoaddLsf(resampledLsf, [rr[0] for rr in resampledRange], [rr[1] for rr in resampledRange])

        return Struct(
            wavelength=combination.wavelength,
            flux=combination.flux,
            sky=combination.sky,
            covar=covar,
            mask=combination.mask,
            covar2=covar2,
            lsf=lsf,
        )

    def combineResampled(
        self, spectra: list[PfsFiberArray], weights: list[np.ndarray], norm: np.ndarray | None = None
    ) -> Struct:
        """Combine spectra that have already been resampled to common
        wavelengths

        Parameters
        ----------
        spectra : `list` of `pfs.datamodel.PfsFiberArray`
            List of spectra to combine for each visit. These must already be
            resampled to a common wavelength array.
        weights : `list` of `numpy.ndarray` of `float`
            List of weights for each spectrum.
        norm : `numpy.ndarray` of `float`, optional
            Normalization to apply to each spectrum before combining. If
            not specified, no normalization is applied.

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Wavelength array for combined spectrum.
        flux : `numpy.ndarray` of `float`
            Flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        variance : `numpy.ndarray` of `float`
            Variance for combined spectrum.
        good : `numpy.ndarray` of `bool`
            Good-pixel mask for combined spectrum.
        """
        if norm is None:
            norm = np.ones(len(spectra), dtype=float)

        archetype = spectra[0]
        length = archetype.length
        mask = np.zeros(length, dtype=archetype.mask.dtype)
        flux = np.zeros(length, dtype=archetype.flux.dtype)
        sky = np.zeros(length, dtype=archetype.sky.dtype)
        variance = np.zeros(length, dtype=archetype.covar.dtype)
        sumWeights = np.zeros(length, dtype=archetype.flux.dtype)

        for ii, ss in enumerate(spectra):
            with np.errstate(invalid="ignore", divide="ignore"):
                good = ((ss.mask & ss.flags.get(*self.config.mask)) == 0) & (ss.variance > 0)
                wt = norm[ii]*weights[ii][good]
                flux[good] += ss.flux[good]*wt
                sky[good] += ss.sky[good]*wt
                mask[good] |= ss.mask[good]
                variance[good] += ss.variance[good]*wt**2
                sumWeights[good] += wt

        good = sumWeights > 0
        flux[good] /= sumWeights[good]
        sky[good] /= sumWeights[good]
        variance[good] /= sumWeights[good]**2

        return Struct(
            wavelength=archetype.wavelength, flux=flux, sky=sky, mask=mask, variance=variance, good=good
        )
