from collections.abc import Mapping
from typing import Dict, Iterator, Iterable, List, Type

import astropy.io.fits
import numpy as np
import yaml
from astropy.io.fits import BinTableHDU, Column, HDUList, ImageHDU
from pfs.datamodel.drp import PfsSingleNotes, PfsSingle, PfsObjectNotes, PfsObject
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.pfsConfig import TargetType
from pfs.datamodel.pfsTable import PfsTable
from pfs.datamodel.target import Target
from pfs.drp.stella.datamodel.fluxTable import FluxTable

from .pfsFiberArray import PfsFiberArray

__all__ = ["PfsTargetSpectra", "PfsCalibratedSpectra", "PfsObjectSpectra"]


class PfsTargetSpectra(Mapping):
    """A collection of `PfsFiberArray` indexed by target"""

    PfsFiberArrayClass: Type[PfsFiberArray]  # Subclasses must override
    NotesClass: Type[PfsTable]  # Subclasses must override

    def __init__(self, spectra: Iterable[PfsFiberArray]):
        super().__init__()
        self.spectra: Dict[Target, PfsFiberArray] = {spectrum.target: spectrum for spectrum in spectra}

    def __getitem__(self, target: Target) -> PfsFiberArray:
        """Retrieve spectrum for target"""
        return self.spectra[target]

    def __iter__(self) -> Iterator[Target]:
        """Return iterator over targets in container"""
        return iter(self.spectra)

    def __len__(self) -> int:
        """Return length of container"""
        return len(self.spectra)

    def __contains__(self, target: Target) -> bool:
        """Return whether target is in container"""
        return target in self.spectra

    @classmethod
    def readFits(cls, filename: str) -> "PfsTargetSpectra":
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        spectra = []
        with astropy.io.fits.open(filename) as fits:
            targetHdu = fits["TARGET"].data
            targetFluxHdu = fits["TARGETFLUX"].data
            observationsHdu = fits["OBSERVATIONS"].data

            wavelength = fits["SPECTRA"].data["wavelength"]
            flux = fits["SPECTRA"].data["flux"]
            mask = fits["SPECTRA"].data["mask"]
            sky = fits["SPECTRA"].data["sky"]
            covar = [cv.reshape((3, len(wl))) for cv, wl in zip(fits["SPECTRA"].data["covar"], wavelength)]

            covar2Hdu = fits["COVAR2"].data if "COVAR2" in fits else None
            metadataHdu = fits["METADATA"].data
            fluxTableHdu = fits["FLUXTABLE"].data
            notesTable = cls.NotesClass.readHdu(fits)

            for ii, row in enumerate(targetHdu):
                targetId = row["targetId"]
                select = targetFluxHdu.targetId == targetId
                fiberFlux = dict(
                    zip(
                        ("".join(np.char.decode(ss.astype("S"))) for ss in targetFluxHdu.filterName[select]),
                        targetFluxHdu.fiberFlux[select],
                    )
                )
                target = Target(
                    row["catId"],
                    row["tract"],
                    "".join(row["patch"]),
                    row["objId"],
                    row["ra"],
                    row["dec"],
                    TargetType(row["targetType"]),
                    fiberFlux=fiberFlux,
                )

                select = observationsHdu.targetId == targetId
                observations = Observations(
                    observationsHdu.visit[select],
                    ["".join(np.char.decode(ss.astype("S"))) for ss in observationsHdu.arm[select]],
                    observationsHdu.spectrograph[select],
                    observationsHdu.pfsDesignId[select],
                    observationsHdu.fiberId[select],
                    observationsHdu.pfiNominal[select],
                    observationsHdu.pfiCenter[select],
                )

                metadataRow = metadataHdu[ii]
                assert metadataRow["targetId"] == targetId

                metadata = yaml.load(
                    # This complicated conversion is required in order to preserve the newlines
                    "".join(np.char.decode(metadataRow["metadata"].astype("S"))),
                    Loader=yaml.SafeLoader,
                )
                flags = MaskHelper.fromFitsHeader(metadata, strip=True)

                fluxTableRow = fluxTableHdu[ii]
                assert fluxTableRow["targetId"] == targetId
                fluxTable = FluxTable(
                    fluxTableRow["wavelength"],
                    fluxTableRow["flux"],
                    fluxTableRow["error"],
                    fluxTableRow["mask"],
                    flags,
                )

                notes = cls.PfsFiberArrayClass.NotesClass(
                    **{col.name: notesTable[col.name][ii] for col in notesTable.schema}
                )

                spectrum = cls.PfsFiberArrayClass(
                    target,
                    observations,
                    wavelength[ii],
                    flux[ii],
                    mask[ii],
                    sky[ii],
                    covar[ii],
                    covar2Hdu[ii] if covar2Hdu is not None else [],
                    flags,
                    metadata,
                    fluxTable,
                    notes,
                )
                spectra.append(spectrum)

        return cls(spectra)

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        fits = HDUList()

        targetId = np.arange(len(self), dtype=np.int16)
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column("catId", "J", array=[target.catId for target in self]),
                    Column("tract", "J", array=[target.tract for target in self]),
                    Column("patch", "PA()", array=[target.patch for target in self]),
                    Column("objId", "K", array=[target.objId for target in self]),
                    Column("ra", "D", array=[target.ra for target in self]),
                    Column("dec", "D", array=[target.dec for target in self]),
                    Column("targetType", "I", array=[int(target.targetType) for target in self]),
                ],
                name="TARGET",
            )
        )

        numFluxes = sum(len(target.fiberFlux) for target in self)
        targetFluxIndex = np.empty(numFluxes, dtype=np.int16)
        filterName: List[str] = []
        fiberFlux = np.empty(numFluxes, dtype=np.float32)
        start = 0
        for tt, target in zip(targetId, self):
            num = len(target.fiberFlux)
            stop = start + num
            targetFluxIndex[start:stop] = tt
            filterName += list(target.fiberFlux.keys())
            fiberFlux[start:stop] = np.array(list(target.fiberFlux.values()))
            start = stop

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetFluxIndex),
                    Column("filterName", "PA()", array=filterName),
                    Column("fiberFlux", "E", array=fiberFlux),
                ],
                name="TARGETFLUX",
            )
        )

        numObservations = sum(len(ss.observations) for ss in self.values())
        observationsIndex = np.empty(numObservations, dtype=np.int16)
        visit = np.empty(numObservations, dtype=np.int32)
        arm: List[str] = []
        spectrograph = np.empty(numObservations, dtype=np.int16)
        pfsDesignId = np.empty(numObservations, dtype=np.int64)
        fiberId = np.empty(numObservations, dtype=np.int32)
        pfiNominal = np.empty((numObservations, 2), dtype=float)
        pfiCenter = np.empty((numObservations, 2), dtype=float)
        start = 0
        for tt, spectrum in zip(targetId, self.values()):
            observations = spectrum.observations
            num = len(observations)
            stop = start + num
            observationsIndex[start:stop] = tt
            visit[start:stop] = observations.visit
            arm += list(observations.arm)
            spectrograph[start:stop] = observations.spectrograph
            pfsDesignId[start:stop] = observations.pfsDesignId
            fiberId[start:stop] = observations.fiberId
            pfiNominal[start:stop] = observations.pfiNominal
            pfiCenter[start:stop] = observations.pfiCenter
            start = stop

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=observationsIndex),
                    Column("visit", "J", array=visit),
                    Column("arm", "PA()", array=arm),
                    Column("spectrograph", "I", array=spectrograph),
                    Column("pfsDesignId", "K", array=pfsDesignId),
                    Column("fiberId", "J", array=fiberId),
                    Column("pfiNominal", "2D", array=pfiNominal),
                    Column("pfiCenter", "2D", array=pfiCenter),
                ],
                name="OBSERVATIONS",
            )
        )

        # Inputs may have different lengths, so we write everything as a giant table
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("wavelength", "PD()", array=[np.array(ss.wavelength) for ss in self.values()]),
                    Column("flux", "PD()", array=[np.array(ss.flux) for ss in self.values()]),
                    Column("mask", "PJ()", array=[np.array(ss.mask) for ss in self.values()]),
                    Column("sky", "PD()", array=[np.array(ss.sky) for ss in self.values()]),
                    Column("covar", "PD()", array=[np.array(ss.covar).flatten() for ss in self.values()]),
                ],
                name="SPECTRA",
            )
        )

        haveCovar2 = [spectrum.covar2 is not None for spectrum in self.values()]
        if len(set(haveCovar2)) == 2:
            raise RuntimeError("covar2 must be uniformly populated")
        if any(haveCovar2):
            fits.append(ImageHDU(data=[spectrum.covar2 for spectrum in self.values()], name="COVAR2"))

        # Metadata table
        metadata: List[str] = []
        for spectrum in self.values():
            md = spectrum.metadata.copy()
            md.update(spectrum.flags.toFitsHeader())
            metadata.append(yaml.dump(md))
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column("metadata", "PA()", array=metadata),
                ],
                name="METADATA",
            )
        )

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column(
                        "wavelength",
                        "PD()",
                        array=[
                            spectrum.fluxTable.wavelength if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "flux",
                        "PD()",
                        array=[
                            spectrum.fluxTable.flux if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "error",
                        "PD()",
                        array=[
                            spectrum.fluxTable.error if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "mask",
                        "PJ()",
                        array=[
                            spectrum.fluxTable.mask if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                ],
                name="FLUXTABLE",
            )
        )

        notes = self.NotesClass.empty(len(self))
        for ii, spectrum in enumerate(self.values()):
            notes.setRow(ii, **spectrum.notes.getDict())
        notes.writeHdu(fits)

        with open(filename, "wb") as fd:
            fits.writeto(fd)


class PfsCalibratedNotesTable(PfsTable):
    """Table of notes for PfsCalibratedSpectra"""
    schema = PfsSingleNotes.schema
    fitsExtName = "NOTES"


class PfsCalibratedSpectra(PfsTargetSpectra):
    """A collection of PfsSingle indexed by target"""
    PfsFiberArrayClass = PfsSingle
    NotesClass = PfsCalibratedNotesTable


class PfsObjectNotesTable(PfsTable):
    """Table of notes for PfsObjectSpectra"""
    schema = PfsObjectNotes.schema
    fitsExtName = "NOTES"


class PfsObjectSpectra(PfsTargetSpectra):
    """A collection of PfsObject indexed by target"""
    PfsFiberArrayClass = PfsObject
    NotesClass = PfsObjectNotesTable
