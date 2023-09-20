import datetime
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import unicodedata

import psycopg2
import pandas as pd
import warnings

import astropy.units
import astropy.coordinates

import lsst.afw.image as afwImage
from pfs.datamodel import PfsDesign
from pfs.datamodel.pfsConfig import FiberStatus, TargetType
from pfs.utils.coordinates.CoordTransp import CoordinateTransform

from lsst.ip.isr import AssembleCcdTask
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

__all__ = ["raDecStrToDeg", "makeDither", "makeCobraImages", "makeSkyImageFromCobras",
           "plotSkyImageOneCobra", "plotSkyImageFromCobras", "calculateOffsets", "offsetsAsQuiver",
           "getWindowedSpectralRange", "getWindowedFluxes", "showDesignPhotometry",
           "estimateExtinction", "plotVisitImage", "getGuideOffset", "showGuiderOffsets"]


def raDecStrToDeg(ra, dec):
    """Convert ra, dec as hexadecimal strings to degrees"""

    return (astropy.coordinates.Angle(ra, unit=astropy.units.h).degree,
            astropy.coordinates.Angle(dec, unit=astropy.units.deg).degree)


class Dither:
    """Information associated with a dithered PFS exposure"""
    def __init__(self, dataId, ra, dec, pa, fluxes, pfsConfig, md, drot=(None, None)):
        self.visit = dataId["visit"]
        self.dataId = dataId.copy()
        self.ra = ra
        self.dec = dec
        self.pa = pa
        self.fluxes = fluxes
        self.pfsConfig = pfsConfig
        self.md = md
        self.drot = drot


def concatenateDithers(butler, dithers):
    d0 = dithers[0]
    for d in dithers[1:]:
        assert d.visit == d0.visit
        assert d.ra == d0.ra
        assert d.dec == d0.dec
        assert d.pa == d0.pa

    n = sum([len(d.fluxes) for d in dithers])
    fluxes = np.empty(n)
    i = 0
    for d in dithers:
        n = len(d.fluxes)
        fluxes[i:i + n] = d.fluxes
        i += n
    fluxes = fluxes

    pfsConfig = butler.get("pfsConfig", d0.dataId)
    keep = np.zeros(len(pfsConfig), dtype=bool)

    for d in dithers:
        keep = np.logical_or(keep, pfsConfig.spectrograph == d.dataId["spectrograph"])

    pfsConfig = pfsConfig[np.logical_and(keep, pfsConfig.targetType != TargetType.ENGINEERING)]

    return Dither(d0.dataId, d0.ra, d0.dec, d0.pa, fluxes, pfsConfig, d0.md)


def getWindowedSpectralRange(row0, nrows, butler=None, dataId=None, detectorMap=None, pfsConfig=None):
    """Return the range of wavelengths available in all fibres for a windowed read

    row0: first row of windowed read
    nrows: number of rows in window
    butler: butler needed to read detectorMap/pfsConfig if not provided
    dataId: descriptor for detectorMap/pfsConfig if not provided
    detectorMap: DetectorMap for data; if None retrieved from butler
    pfsConfig: PfsConfig for data; if None retrieved from butler

    returns:
      lambda_min, lambda_max
    """
    if detectorMap is None or pfsConfig is None:
        assert butler is not None and dataId is not None
        if detectorMap is None:
            detectorMap = butler.get("detectorMap", dataId)
        if pfsConfig is None:
            pfsConfig = butler.get("pfsConfig", dataId)

    lmin = max(detectorMap.findWavelength(pfsConfig.fiberId,
                                          np.full(len(pfsConfig), float(row0))))
    lmax = min(detectorMap.findWavelength(pfsConfig.fiberId,
                                          np.full(len(pfsConfig), float(row0 + nrows - 1))))

    return lmin, lmax


def getPFSA(dirName, dataId):
    """Lookup a PFSA file that was written this year"""
    today = datetime.date.today()
    num = dict(b=1, r=2, n=3, m=4)[dataId['arm']]
    files = glob.glob(f"{dirName}/{today.year}*/sps/PFSA%(visit)06d%(spectrograph)d{num}.fits" % dataId)
    if files:
        return files[0]
    else:
        raise RuntimeError(f"Unable to find PFSA file for {dataId}")


def getWindowedFluxes(butler, dataId, fiberTraces=None, darkVariance=30,
                      camera=None, useButler=True, usePfsArm=False, visit0=0, **kwargs):
    """Return an estimate of the median flux in each fibre

    butler: a butler pointing at the data
    dataId: a dict specifying the desired data
    fiberTraces: a dict indexed by arm containing the appropriate FiberTrace objects
                 If None, read and construct the FiberTrace
    useButler: if False bypass the butler for raw/pfsConfig files; caveat emptor
    usePfsArm: use the pfsArm file rather than starting with the raw
    visit0: If non-zero, get the detectorMap from this visit
    kwargs: overrides for dataId
    """
    dataId = dataId.copy()
    dataId.update(kwargs)
    if not visit0:
        visit0 = dataId["visit"]

    if usePfsArm:
        md = butler.get("raw_md", dataId)
        pfsConfig = butler.get("pfsConfig", dataId)
        spectra = butler.get("pfsArm", dataId)

        pfsConfig = pfsConfig[pfsConfig.spectrograph == dataId["spectrograph"]]

        spectra = spectra[pfsConfig.targetType != TargetType.ENGINEERING]
        pfsConfig = pfsConfig[pfsConfig.targetType != TargetType.ENGINEERING]
    else:
        detectorMap = butler.get("detectorMap", dataId, visit=visit0)

        if fiberTraces is None:
            fiberProfiles = butler.get("fiberProfiles", dataId)
            detectorMap = butler.get("detectorMap", dataId)

            fiberTrace = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)
        else:
            fiberTrace = fiberTraces[dataId["arm"]]

        if useButler:
            pfsConfig = butler.get("pfsConfig", dataId)
            exp = butler.get("raw", dataId, filter=dataId['arm']).convertF()
        else:
            if camera is None:
                camera = butler.get("camera")

            fileName = getPFSA("/data/raw", dataId)
            raw = afwImage.ImageF.readFits(fileName)
            md = afwImage.readMetadata(fileName)

            exp = afwImage.makeExposure(afwImage.makeMaskedImage(raw))
            exp.setMetadata(md)
            exp.setDetector(camera["%(arm)s%(spectrograph)d" % dataId])

            pfsDesignId = exp.getMetadata()["W_PFDSGN"]

            pfsConfig = PfsDesign.read(pfsDesignId, "/data/pfsDesign")

        pfsConfig = pfsConfig[np.logical_and(pfsConfig.spectrograph == dataId["spectrograph"],
                                             pfsConfig.targetType != TargetType.ENGINEERING)]

        #
        # Instantiate our tasks (we could do this once, but it's fast)
        #
        config = AssembleCcdTask.ConfigClass()
        config.doTrim = True
        assembleTask = AssembleCcdTask(config=config)

        config = ExtractSpectraTask.ConfigClass()
        config.validate()
        extractSpectra = ExtractSpectraTask(config=config)
        #
        # Ready for the ISR
        #
        md = exp.getMetadata()
        row0, row1 = md["W_CDROW0"], md["W_CDROWN"]

        maskVal = exp.mask.getPlaneBitMask(["SAT", "NO_DATA"])
        exp.mask.array[0:row0] = maskVal
        exp.mask.array[row1 + 1:] = maskVal
        exp.image.array[exp.mask.array != 0] = np.NaN  # not 0; we're going to use np.nanmedian later

        for amp in exp.getDetector():
            exp[amp.getRawBBox()].image.array -= \
                np.nanmedian(exp[amp.getRawHorizontalOverscanBBox()].image.array)

        exp = assembleTask.assembleCcd(exp)
        exp.variance = exp.image
        exp.variance.array += darkVariance  # need a floor to the noise to reduce the b arm

        spectra = extractSpectra.run(exp.maskedImage, fiberTrace, detectorMap).spectra.toPfsArm(dataId)

    spectra.flux[spectra.mask != 0] = np.NaN

    return pfsConfig, md, np.nanmedian(spectra.flux, axis=1)


def makeDither(butler, dataId, lmin=665, lmax=700, targetType=None, fiberTraces=None,
               useButler=True, visit0=0, usePfsArm=False, camera=None):
    """Return an element of the dithers[] array that's passed to makeCobraImages"""

    pfsConfig, md, fluxes = getWindowedFluxes(butler, dataId, fiberTraces, useButler=useButler, visit0=visit0,
                                              usePfsArm=usePfsArm, camera=camera)
    fluxes[np.isnan(pfsConfig.pfiNominal.T[0])] = np.NaN

    ra, dec = raDecStrToDeg(md['RA_CMD'], md['DEC_CMD'])
    pa = md['INST-PA']

    return Dither(dataId, ra, dec, pa, fluxes, pfsConfig, md)


def makeCobraImages(dithers, side=4, pixelScale=0.025, R=50, fiberIds=None,
                    usePFImm=False, subtractBkgd=True, setUnimagedPixelsToNaN=False,
                    useInitialPointing=False, skipFirstDither=False,
                    extinction=None,
                    icrosstalk=None):
    """
    dithers: a list of Dithers (carrying (visit, ra, dec, fluxes, pfsConfig, md))
             where (ra, dec) are the boresight pointing, fluxes are the fluxes in the fibres,
             and md is the image metadata
    side: length of side of postage stamp image, arcsec
    pixelScale: size of pixels, arcsec
    R: radius of fibre, microns
    fiberIds: only make images for these fibers
    skipFirstDither: use first dither for geometry, but omit its photons from the raster image
    usePFImm: make image in microns on PFI.  In this case, replace "arcsec" in side/pixelScale by "micron"
    subtractBkgd: subtract an estimate of the background
    extinction: dict of extinction values for each visit in dithers

    Returns:
       images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
       extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)
    """

    if len(dithers) == 0:
        raise RuntimeError("dithers array is empty")

    if fiberIds is not None:
        try:
            fiberIds[0]
        except TypeError:
            fiberIds = [fiberIds]

    if side <= 0:
        if not usePFImm:
            raise RuntimeError("Code to calculate size only works in mm; sumimasen")

        for i, d in enumerate(dithers):
            ra = d.ra
            dec = d.dec
            pa = d.pa
            md = d.md
            pfsConfig = d.pfsConfig

            cosDec = np.cos(np.deg2rad(dec))
            boresight = [[ra], [dec]]

            altitude = md['ALTITUDE']
            utc = f"{md['DATE-OBS']} {md['UT']}"

            try:
                x, y = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec])),
                                           mode="sky_pfi", za=90.0 - altitude,
                                           pa=pa, cent=boresight, time=utc)[0:2]
            except ValueError:          # not on sky; no valid boresight
                x, y = np.zeros_like(ra), np.zeros_like(dec)

            if i == 0:
                x0, y0 = x, y

            dx, dy = 1e3*(x - x0), 1e3*(y - y0)   # microns

            xmin = np.nanmin(dx - R/pixelScale)
            xmax = np.nanmax(dx + R/pixelScale)
            ymin = np.nanmin(dy - R/pixelScale)
            ymax = np.nanmax(dy + R/pixelScale)

            print(xmin, xmax, ymin, ymax)

    R_micron = R                        # R in microns, before mapping to arcsec on PFI
    size = None
    altitude0 = None
    for d in dithers:
        fluxes = d.fluxes
        ra = d.ra
        dec = d.dec
        pa = d.pa
        md = d.md
        pfsConfig = d.pfsConfig

        if subtractBkgd:
            bkgd = np.nanpercentile(fluxes, [5])[0]
            fluxes = fluxes - bkgd

        if icrosstalk is not None:
            fluxes = icrosstalk@np.where(np.isfinite(fluxes), fluxes, 0)

        fluxes[d.pfsConfig.fiberStatus != FiberStatus.GOOD] = np.NaN

        if extinction is not None:
            fluxes = fluxes*10**(0.4*extinction.get(d.visit, 0.0))

        cosDec = np.cos(np.deg2rad(dec))
        boresight = [[ra], [dec]]

        altitude = md['ALTITUDE']
        utc = f"{md['DATE-OBS']} {md['UT']}"

        #
        # Use this initial altitude/utc for coordinate transforms?
        #
        if useInitialPointing:
            if altitude0 is None:
                altitude0 = altitude
                utc0 = utc
                pa0 = pa

            altitude = altitude0
            utc = utc0
            pa = pa0

        hexapod_sx, hexapod_sy = md["W_M2OFF1"], md["W_M2OFF2"]
        hexapod_sx0, hexapod_sy0 = -1.6, -2.5  # as per Yuki Moritani

        hexapod_sx -= hexapod_sx0
        hexapod_sy -= hexapod_sy0

        if False:
            print(f"{d.visit} ({3600*(d.ra - dithers[0].ra):5.1f}, {3600*(d.dec - dithers[0].dec):5.1f}) "
                  f"{pa}, {hexapod_sx}, {hexapod_sy}")

        if usePFImm:
            try:
                x, y = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec])),
                                           mode="sky_pfi", za=90.0 - altitude,
                                           pa=pa, cent=boresight, time=utc)[0:2]
            except ValueError:          # not on sky; no valid boresight
                x, y = np.zeros_like(ra), np.zeros_like(dec)

            x += hexapod_sx
            y += hexapod_sy

            R = np.full(len(pfsConfig), R_micron)
        else:
            # The sky_pfi and pfi_sky transforms don't roundtrip, so we can't trivially
            # calculate the fibre radius in arcsec by transformin from/to the sky.  We could use
            # sky_pfi, then pfi_sky twice to measure the radius, but instead we can guess a value in arcsec,
            # find what it is in microns, and scale.  I.e. sky_pfi twice.

            R_asec = 0.5  # approximate radius in arcsec; the actual value doesn't matter
            x, y = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec])),
                                       mode="sky_pfi", za=90.0 - altitude,
                                       pa=pa, cent=boresight, time=utc)[0:2]
            x2, y2 = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec + R_asec/3600])),
                                         mode="sky_pfi", pa=pa, za=90.0 - altitude,
                                         cent=boresight, time=utc)[0:2]
            R = np.hypot(x - x2, y - y2)*1e3   # microns corresponding to R_asec
            R = np.full(len(pfsConfig), R_asec*R_micron/R)  # asec

            x, y = np.full_like(x, ra), np.full_like(x, dec)

        if size is None:                # first time through the loop
            x0, y0 = x, y               # offset relative to first dither

            size = int(side/pixelScale + 1)
            if size%2 == 0:
                size += 1
            hsize = size//2

            extent = np.array((-hsize - 0.5, hsize + 0.5, -hsize - 0.5, hsize + 0.5))*pixelScale

            I, J = np.meshgrid(np.arange(0, size), np.arange(0, size))

            nCobra = len(pfsConfig)
            images = np.zeros((nCobra, size, size))
            weights = np.zeros_like(images)

            visitImage = np.full_like(images[0], 0)

            if skipFirstDither:
                continue

        if usePFImm:
            dx, dy = 1e3*(x - x0), 1e3*(y - y0)   # microns
        else:
            dx, dy = 3600*(x - x0)*cosDec, 3600*(y - y0)  # arcsec

        xc, yc = hsize + dx/pixelScale, hsize + dy/pixelScale

        for i, (R, fid) in enumerate(zip(R, pfsConfig.fiberId)):
            if fiberIds and fid not in fiberIds:
                continue
            r = np.hypot(I - yc[i], J - xc[i])
            r = np.where(np.isfinite(r), r, 10*R/pixelScale)  # avoid annoying warning on < from np.where
            illum = np.where(r < R/pixelScale, 1.0, 0.0)
            images[i, I, J] += illum*fluxes[i]
            weights[i, I, J] += illum
            ii = i                      # index for visitImage

        r = np.hypot(I - yc[ii], J - xc[ii])
        visitImage[I, J] = np.where(r < R/pixelScale, d.visit, visitImage[I, J])

    images /= np.where(weights == 0, 1, weights)
    if setUnimagedPixelsToNaN:
        images[weights == 0] = np.NaN
        visitImage[weights[ii] == 0] = np.NaN

    return images, extent, visitImage


def makeSkyImageFromCobras(pfsConfig, images, pixelScale, setUnimagedPixelsToNaN=True,
                           usePFImm=False, swaps=[], figure=None):
    """
    usePFImm: make image in microns on PFI.  In this case, replace "arcsec" in side/pixelScale by "micron"
    swaps: list of pairs of switched fibres
    """
    if usePFImm:
        cosDec = 1.0
        x, y = pfsConfig.pfiNominal.T
    else:
        cosDec = np.cos(np.deg2rad(np.nanmean(pfsConfig.dec)))
        x, y = pfsConfig.ra, pfsConfig.dec

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    stampsize = images[0].shape[0]      # stamps of each cobra's images are stampsize x stampsize

    if usePFImm:
        scale = 1e3/pixelScale          # pixel/mm
    else:
        scale = 3600/pixelScale         # pixel/deg

    border = 0.55*stampsize/scale       # needed extra width (on the PFI image)

    xmin -= border/cosDec
    xmax += border/cosDec
    ymin -= border
    ymax += border

    imsize = int(max((xmax - xmin)*cosDec, ymax - ymin)*scale) + 2*stampsize
    xsize = (imsize - 1)/cosDec/scale
    xmin = (xmin + xmax)/2 - xsize/2   # beware x=0 == 360!
    xmax = xmin + xsize

    ysize = (imsize - 1)/scale
    ymin = (ymin + ymax)/2 - ysize/2
    ymax = ymin + ysize

    pfiIm = np.full((imsize, imsize), 0.0)
    weights = np.zeros_like(pfiIm)
    pfiImMask = np.zeros_like(pfiIm, dtype=bool)

    nCobra = len(pfsConfig)              # sum(pfsConfig.targetType != TargetType.ENGINEERING)
    stampOffset = np.empty((nCobra, 2))  # offset of stamp from its true position due to zc = int(z + 0.5)
                                         # noqa: E114, E116     rounding to snap to grid (pass for flake8)

    for _i in range(nCobra):
        i = _i
        for swap in swaps:
            if pfsConfig.fiberId[_i] in swap:
                print(f"Switching fiberId {pfsConfig.fiberId[_i]}, cobraId {_i}")
                if pfsConfig.fiberId[_i] == swap[0]:
                    i = pfsConfig.selectFiber(swap[1])
                else:
                    i = pfsConfig.selectFiber(swap[0])

                break

        xx, yy = np.array(((x[_i] - xmin)*cosDec, y[_i] - ymin))*scale  # N.b. We use _i here
        if not np.isfinite(xx + yy):
            if pfsConfig.targetType[_i] not in (TargetType.ENGINEERING, TargetType.UNASSIGNED):
                print(f"pfsConfig 0x{pfsConfig.pfsDesignId:x}"
                      f" coordinates for fiberId {pfsConfig.fiberId[i]} are NaN: ({xx}, {yy})."
                      f" ObjectType is {str(TargetType(pfsConfig.targetType[_i]))};"
                      )
            continue
        xc = int(xx + 0.5)
        yc = int(yy + 0.5)

        stampOffset[i] = (xc/(cosDec*scale) + xmin - x[i], yc/scale + ymin - y[i])

        x0, x1 = xc - stampsize//2, xc + stampsize//2 + 1
        y0, y1 = yc - stampsize//2, yc + stampsize//2 + 1
        try:
            good = np.isfinite(images[i])
            pfiIm[y0:y1, x0:x1][good] += images[i][good]
            weights[y0:y1, x0:x1][good] += 1
            pfiImMask[y0:y1, x0:x1][good] = True
        except (IndexError, ValueError):
            print(f"fiberId {pfsConfig.fiberId[i]} doesn't fit in image")

    if setUnimagedPixelsToNaN:
        pfiIm[pfiImMask == False] = np.NaN  # noqa: E712  "is False" doesn't produce an array of bool

    pfiIm /= np.where(weights == 0, 1, weights)

    raPixsize  = (xmax - xmin)/(imsize - 1)  # noqa E221: size of pixels in ra direction
    decPixsize = (ymax - ymin)/(imsize - 1)  # :          and in dec direction
    extent = (xmin - raPixsize/2, xmax + raPixsize/2, ymin - decPixsize/2, ymax + decPixsize/2)

    return pfiIm, extent, stampOffset


def plotSkyImageOneCobra(fiberId, pfsConfig, images, extent_CI, usePFImm=False, vmin=None, vmax=None,
                         gfm=None):
    """Plot the synthesised image of the sky from dithered spectra
    fiberId: desired cobra
    pfsConfig: PFI configuration
    images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
    extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)
    vmin, vmax: limits for scaling image
    gfm: `pfs.utils.fiberids.FiberIds` the Grand Fiber Map for looking up the cobraId

    images, extent_CI are as returned by makeCobraImages
    """
    cmap = plt.matplotlib.cm.get_cmap().copy()
    cmap.set_bad(color='black')

    ims = plt.imshow(images[pfsConfig.selectFiber(fiberId)], origin='lower', interpolation='none',
                     extent=extent_CI, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_aspect(1)

    plt.plot(0, 0, '+', color='white')

    if usePFImm:
        plt.xlabel(r"$\Delta x$ ($\mu m$)")
        plt.ylabel(r"$\Delta y$ ($\mu m$)")
    else:
        plt.xlabel(r"$\Delta\alpha$ (arcsec)")
        plt.ylabel(r"$\Delta\delta$ (arcsec)")

    title = f"fiberId={fiberId}"
    if gfm:
        title += f" cobraId = {gfm.cobraId[gfm.fiberId == fiberId][0]}"

    plt.title(title)

    return ims


def plotSkyImageFromCobras(pfsConfig, pfiIm, extent, stampOffset, usePFImm=False, showNominalPosition=True,
                           vmin=None, vmax=None, norm=None):
    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])  # == 1/cosDec

    if usePFImm:
        x, y = pfsConfig.pfiNominal.T
    else:
        x, y = pfsConfig.ra, pfsConfig.dec

    cmap = plt.matplotlib.cm.get_cmap().copy()
    cmap.set_bad(color='black')

    ims = plt.imshow(pfiIm, origin='lower', interpolation='none', cmap=cmap,
                     extent=extent, vmin=vmin, vmax=vmax, norm=norm)
    if showNominalPosition:
        plt.plot(x + stampOffset[:, 0], y + stampOffset[:, 1], '+', alpha=0.2, color='white')

    plt.gca().set_aspect(aspect)

    if usePFImm:
        plt.xlabel(r"x (mm)")
        plt.ylabel(r"y (mm)")
    else:
        plt.xlabel(r"$\alpha$ (deg)")
        plt.ylabel(r"$\delta$ (deg)")

    return ims


def plotVisitImage(visitImage, extent_CI, usePFImm=False):
    """Plot an image giving the visit numbers which went into a raster
    The visitImage, as returned by makeCobraImages
    extent_CI: (x0, x1, y0, y1) giving the corners of visitImage (as passed to plt.imshow)

    visitImage, extent_CI are as returned by makeCobraImages
    """
    cmap = plt.matplotlib.cm.get_cmap().copy()
    cmap.set_bad(color='black')

    vmin = np.nanmin(visitImage)
    vmax = np.nanmax(visitImage)

    ims = plt.imshow(visitImage, origin='lower', interpolation='none',
                     extent=extent_CI, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_aspect(1)

    plt.plot(0, 0, '+', color='white')

    if usePFImm:
        plt.xlabel(r"$\Delta x$ ($\mu m$)")
        plt.ylabel(r"$\Delta y$ ($\mu m$)")
    else:
        plt.xlabel(r"$\Delta\alpha$ (arcsec)")
        plt.ylabel(r"$\Delta\delta$ (arcsec)")

    return ims


def calculateOffsets(images, extent_CI):
    """Calculate and return the offset of the centroid for each of the images
    images:  ndarray of shape (ncobra, xsize, ysize)
    extent_CI: tuple (x0, x1, y0, y1) giving the corners of the images[n]

    return xoff, yoff in the units specified by extent_CI (the size of the cobra stamps; microns or arcsec)
    """
    x0, x1, y0, y1 = extent_CI
    xs, ys = images[0].shape
    x, y = np.meshgrid(np.linspace(y0, y1, xs), np.linspace(x0, x1, ys))
    #
    # vectorize the calculation, at the expense of memory.  Is it worth it???
    #
    x = np.concatenate(len(images)*[x]).reshape((len(images), xs, ys))
    y = np.concatenate(len(images)*[y]).reshape((len(images), xs, ys))

    weights = np.where(np.isfinite(images), images, 0)
    image_weight = np.nansum(weights, axis=(1, 2))
    xoff = np.full(len(images), np.NaN)
    yoff = np.full_like(xoff, np.NaN)

    good = image_weight > 0
    xoff[good] = np.average(x[good], axis=(1, 2), weights=weights[good])
    yoff[good] = np.average(y[good], axis=(1, 2), weights=weights[good])

    return xoff, yoff, image_weight


def offsetsAsQuiver(pfsConfig, xoff, yoff, usePFImm=False, select=None,
                    quiverLen=None, quiverLabel=None, scale=None, C=None, quiver_args={}):
    """Plot the offsets (as calculated by calculateOffsets) as a quiver

    pfsConfig:    description of targetting
    xoff, yoff:   measured positional offsets, in (microns if usePFImm else arcsec)
    usePFImm :    make plot in PFI mm rather than ra/dec (you may need to scale xoff/yoff)
    select:       only show points for which select is True (if not None)
    quiverLen:    passed to quiverkey (cv.) as "U"
    quiverLabel:  passed to quiverkey (cv.) as "label"
    scale:        passed to quiver (cv.)
    C:            passed to quiver (cv.)
    color:        passed to quiver (cv.)
    quiver_args: dict of extra arguments to pass to quiver
                 (so you could set scale as quiver_args=dict(scale=...))

    Arguments
       quiverLen, quiverLabel, scale, C, color
    are passed to quiver, along with **quiver_args

    Returns the Quiver
    """
    if select is None:
        select = np.ones_like(xoff, dtype=bool)

    if usePFImm:
        cosDec = 1
        x, y = pfsConfig.pfiNominal.T
        if quiverLen is None:
            quiverLen = 50
        if quiverLabel is None:
            quiverLabel = r"$\mu$m"
    else:
        cosDec = np.cos(np.deg2rad(pfsConfig.decBoresight))
        x, y = pfsConfig.ra, pfsConfig.dec
        if quiverLen is None:
            quiverLen = 0.5
        if quiverLabel is None:
            quiverLabel = "arcsec"

    args = [x[select], y[select], xoff[select], yoff[select]]
    if C is not None:
        args.append(C[select])                  # can't just pass None in the "C" slot

    if "alpha" in quiver_args:
        quiver_args.update(alpha=quiver_args["alpha"][select])

    Q = plt.quiver(*args, scale=scale, **quiver_args)

    plt.quiverkey(Q, 0.1, 0.9, quiverLen, f"{quiverLen}{quiverLabel}", coordinates='axes',
                  color='red', labelcolor='red')

    plt.gca().set_aspect(1 if usePFImm else 1/cosDec)

    if usePFImm:
        plt.xlabel(r"x (mm)")
        plt.ylabel(r"y (mm)")
    else:
        plt.xlabel(r"$\alpha$ (deg)")
        plt.ylabel(r"$\delta$ (deg)")

    return Q


#
# Workaround (harmless but annoying) pandas warning telling me to use sqlalchemy to access postgres
#
def pd_read_sql(sql_query: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Execute SQL Query and get Dataframe with pandas"""
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        # create pandas DataFrame from database query
        df = pd.read_sql_query(sql_query, db_conn)
    return df


def estimateExtinction(opdb, visit, magLim=16, zeroPoint=29.06):
    """Estimate the extinction for a given visit, using Gaia stars detected by the AG code
    opdb: connection to the opdb
    visit: desired visit
    magLim: magnitude limit for Gaia stars to use
    zeroPoint: estimated zero point for AG photometry (if this is wrong, the extinction will be wrong
               by the same amount)
    """
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT
                agc_exposure_id
            FROM agc_exposure
            JOIN sps_exposure ON sps_exposure.pfs_visit_id = agc_exposure.pfs_visit_id
            WHERE
                sps_exposure.pfs_visit_id = {visit} AND
                agc_exposure.taken_at BETWEEN sps_exposure.time_exp_start AND sps_exposure.time_exp_end
            ORDER BY agc_exposure ASC
            ''', opdb)

    amin, amax = np.min(tmp.agc_exposure_id), np.max(tmp.agc_exposure_id)

    if np.isnan(amin + amax):
        return 0.0

    with opdb:
        tmp = pd_read_sql(f'''
        SELECT
            pfs_visit_id, agc_exposure.agc_exptime, agc_exposure.agc_exposure_id, agc_exposure.taken_at,
            agc_match.guide_star_id, image_moment_00_pix, pfs_design_agc.guide_star_magnitude,
            pfs_design_agc.guide_star_color
        FROM agc_exposure
        JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
        JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id AND
                          agc_match.agc_camera_id = agc_data.agc_camera_id AND
                          agc_match.spot_id = agc_data.spot_id
        JOIN pfs_design_agc ON pfs_design_agc.guide_star_id = agc_match.guide_star_id
        WHERE
            pfs_design_agc.passband = 'g_gaia' and
            agc_exposure.agc_exposure_id BETWEEN {amin} AND {amax}
        ''', opdb)
    #
    # Set an array, it, which is the index into the possible timestamps
    # We use this to add NaN entries for guide stars which are only sometimes detected; there
    # must be a pandagenic way to do this
    #
    it = np.empty(len(tmp), dtype=int)
    taken_ats = np.array(sorted(list(set(tmp.taken_at))))
    for i, t in enumerate(taken_ats):
        it[tmp.taken_at == t] = i

    tmp["it"] = it

    guide_star_ids = sorted(list(set(tmp.guide_star_id)))
    mag = zeroPoint - 2.5*np.log10(tmp.image_moment_00_pix)

    mm = np.full((len(guide_star_ids), len(taken_ats)), np.NaN)
    for i, gid in enumerate(guide_star_ids):
        ll = tmp.guide_star_id == gid
        if np.mean(tmp.guide_star_magnitude[ll] > magLim):
            continue

        mm[i][tmp.it[ll]] = mag[ll] - np.median(tmp.guide_star_magnitude[ll])

    return 0.0 if len(mm) == 0 else np.nanmedian(mm)


def getGuideOffset(opdb, visit):
    """Estimate the mean guide offset for a given visit
    opdb: connection to the opdb
    visit: desired visit

    Returns:
       dra   mean offset in ra (arcseconds).  N.b. no cos(dec) so can be directly added to ra
       ddec  mean offset in dec (arcseconds)
       df    Pandas data frame for diagnostics, if desired
    """
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT
                agc_exposure_id
            FROM agc_exposure
            JOIN sps_exposure ON sps_exposure.pfs_visit_id = agc_exposure.pfs_visit_id
            WHERE
                sps_exposure.pfs_visit_id = {visit} AND
                agc_exposure.taken_at BETWEEN sps_exposure.time_exp_start AND sps_exposure.time_exp_end
            ORDER BY agc_exposure ASC
            ''', opdb)

    amin, amax = np.min(tmp.agc_exposure_id), np.max(tmp.agc_exposure_id)

    if np.isnan(amin + amax):
        return 0, 0, None

    with opdb:
        tmp = pd_read_sql(f'''
            SELECT
                pfs_visit.pfs_visit_id, agc_exposure.agc_exposure_id, agc_exposure.taken_at,
                guide_ra, guide_dec, guide_delta_ra, guide_delta_dec
            FROM pfs_visit
            JOIN agc_exposure ON agc_exposure.pfs_visit_id = pfs_visit.pfs_visit_id
            JOIN agc_guide_offset ON agc_guide_offset.agc_exposure_id = agc_exposure.agc_exposure_id
            WHERE
                 agc_exposure.agc_exposure_id BETWEEN {amin} AND {amax}
            ''', opdb)

    if len(tmp) == 0:
        return 0, 0, tmp

    return np.nanmean(tmp.guide_delta_ra), np.nanmean(tmp.guide_delta_dec), tmp


def getDitherRaDec(opdb, visit):
    """Return the dither_ra/dec for a visit

    Will be in headers in the great bye-and-bye
    """
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT
                dither_ra, dither_dec, status_sequence_id
            FROM tel_status
            JOIN sps_exposure ON sps_exposure.pfs_visit_id = tel_status.pfs_visit_id
            WHERE
                tel_status.pfs_visit_id = {visit} AND
                tel_status.created_at BETWEEN sps_exposure.time_exp_start AND sps_exposure.time_exp_end
            ORDER BY status_sequence_id ASC
            ''', opdb)

    if len(tmp) == 0:
        return np.NaN, np.NaN, tmp

    if len(set(tmp.dither_ra)) != 1 or len(set(tmp.dither_dec)) != 1:
        print(f"getDitherRaDec: detected multiple dither_XXX values for visit {visit}:\n", tmp)

    return tmp.dither_ra[0], tmp.dither_dec[0], tmp


def getGuideError(opdb, visit):
    """Return the AG's guide error estimate for a visit
    opdb: connection to the opdb
    visit: desired visit

    Returns:  delta(altitude), delta(azimuth), delta(insrot) all in arcseconds
    """

    with opdb:
        tmp = pd_read_sql(f'''
           SELECT
               agc_exposure.pfs_visit_id,
               -- avg(agc_exposure.altitude) AS altitude,
               -- avg(agc_exposure.azimuth) AS azimuth,
               -- avg(agc_exposure.insrot) AS insrot,
               min(guide_delta_insrot) as guide_delta_insrot,
               min(guide_delta_az) as guide_delta_azimuth,
               min(guide_delta_el) as guide_delta_altitude
           FROM agc_exposure
           JOIN agc_guide_offset ON agc_guide_offset.agc_exposure_id = agc_exposure.agc_exposure_id
           JOIN sps_exposure ON sps_exposure.pfs_visit_id = agc_exposure.pfs_visit_id
           WHERE
               agc_exposure.pfs_visit_id = {visit} AND
               agc_exposure.taken_at BETWEEN sps_exposure.time_exp_start AND sps_exposure.time_exp_end
           GROUP BY agc_exposure.pfs_visit_id -- agc_exposure.agc_exposure_id
           ''', opdb)

    tmp.reset_index(drop=True)

    return tmp.guide_delta_altitude[0], tmp.guide_delta_azimuth[0], tmp.guide_delta_insrot[0]


def showGuiderOffsets(opdb, visits, showGuidePath=True, showMeanToEndOffset=False):
    """
    Plot the guide offsets for a set of visits

    N.b. The guider's guide_delta_ra is multiplied by cos(dec) in all of these plots

    opdb:                 a connection to the opdb postgres database
    showGuidePath:        plot dAlpha : dDelta
    showMeanToEndOffset:  plot the difference between the average offset in a visit and the last value

    returns: matplotlib.Figure
    """

    fig, axs = plt.subplots(1 if showGuidePath else 2, 1, sharex=True, squeeze=False,
                            gridspec_kw=dict(hspace=0))
    axs = axs.flatten()

    oldxy = None
    for visit in visits:
        dra, ddec, tmp = getGuideOffset(opdb, visit)

        if tmp is None or len(tmp) == 0:
            continue

        cosDec = np.cos(np.deg2rad(tmp.guide_delta_dec))[0]
        tmp.guide_delta_ra *= cosDec
        dra *= cosDec

        if showGuidePath:
            x, y = tmp.guide_delta_ra.to_numpy(), tmp.guide_delta_dec.to_numpy()
            color = plt.plot(x, y, label=f"{visit}")[0].get_color()
            if oldxy is None:
                plt.plot([x[0]], [y[0]], 'o', color='black', fillstyle='none', zorder=-1)
            else:
                plt.plot([oldxy[0], x[0]], [oldxy[1], y[0]], ':', color=color, alpha=0.5)
            oldxy = [x[-1], y[-1]]
        elif showMeanToEndOffset:
            plt.sca(axs[0])
            plt.plot([np.mean(tmp.taken_at)], tmp.guide_delta_ra.to_numpy()[-1] - dra, 'o', color='black')

            plt.sca(axs[1])
            plt.plot([np.mean(tmp.taken_at)], tmp.guide_delta_dec.to_numpy()[-1] - ddec, 'o', color='black')
        else:
            plt.sca(axs[0])
            color = plt.plot(tmp.taken_at, tmp.guide_delta_ra, 'o')[0].get_color()
            plt.plot([np.mean(tmp.taken_at)], dra, '+', color='black')

            plt.sca(axs[1])
            plt.plot(tmp.taken_at, tmp.guide_delta_dec, 'o', color=color)
            plt.plot([np.mean(tmp.taken_at)], ddec, '+', color='black')

    if showGuidePath:
        if oldxy is not None:
            plt.plot(*oldxy, 'o', color='black')
        axs[0].set_aspect(1)
        plt.xlabel(r"$\Delta\alpha$ (arcsec)")
        plt.ylabel(r"$\Delta\delta$ (arcsec)")
    else:
        plt.sca(axs[0])
        if showMeanToEndOffset:
            plt.axhline(0, ls='-', color="red")
        plt.ylabel(r"$\Delta\alpha$ (arcsec)")

        plt.sca(axs[1])
        if showMeanToEndOffset:
            plt.axhline(0, ls='-', color="red")

        plt.ylabel(r"$\Delta\delta$ (arcsec)")
        plt.xlabel("taken_at")

    (plt.title if showGuidePath else plt.suptitle)(f"visits {visits[0]}:{visits[-1]}")

    return fig


class ShowCobra:
    """Show a cobraId and possibly fiberId on right-click"""

    def __init__(self, ax, pfi, gfm=None, pfsConfig=None):
        self.ax = ax
        self.pfi = pfi
        self.gfm = gfm
        self.pfsConfig = pfsConfig
        self.text = ax.text(0, 0, "", va="bottom", ha="left", color='white', transform=ax.transAxes)
        self.circle = None
        #
        self.__alpha = unicodedata.lookup("GREEK SMALL LETTER alpha")  # used in cursor display string
        self.__delta = unicodedata.lookup("GREEK SMALL LETTER delta")  # used in cursor display string
        #
        # Save the values
        #
        self.cobraId = -1
        self.fiberId = -1
        self.ra = np.NaN
        self.dec = np.NaN
        #
        # For debugging
        #
        self.event = None
        self.exception = None
        self.msg = ""

    def __call__(self, event):
        self.event = event
        if self.circle:
            self.circle.remove()
            self.circle = None

        self.msg = ""
        try:
            if event.button in [3]:
                x, y = event.xdata, event.ydata

                self.fiberId = None
                cobraColor = 'white'
                if self.gfm and self.pfsConfig:
                    xnom, ynom = self.pfsConfig.pfiNominal.T
                    i = np.nanargmin(np.hypot(xnom - x, ynom - y))
                    self.fiberId = self.pfsConfig.fiberId[i]
                    self.cobraId = self.gfm.cobraId[self.gfm.fiberId == self.fiberId][0]
                    x, y = self.pfsConfig.pfiNominal[i]
                    cobraColor = 'red'
                else:                   # find nearest cobra
                    self.cobraId = np.argmin(np.abs(self.pfi.centers - (x + 1j*y))) + 1
                    if False:  # just a guess, so probably worse than useless
                        xy = self.pfi.centers[self.cobraId - 1]
                        x, y = xy.real, xy.imag

                self.msg += f"cobraId {self.cobraId:4}"

                if self.gfm:
                    if self.fiberId is None:
                        self.fiberId = self.gfm.fiberId[self.gfm.cobraId == self.cobraId][0]

                    self.msg += f"  fiberId {self.fiberId:4}"

                    if self.pfsConfig:
                        ll = self.pfsConfig.fiberId == self.fiberId
                        self.ra, self.dec = self.pfsConfig.ra[ll][0], self.pfsConfig.dec[ll][0]
                        self.msg += f" ({self.__alpha},{self.__delta})=({self.ra:7.4f}, {self.dec:7.4f})"

                self.circle = Circle((x, y), 2, facecolor=None, edgecolor=cobraColor, fill=False, alpha=1)
                self.ax.add_patch(self.circle)

        except Exception as e:
            self.exception = e
        else:
            self.exception = None

        self.text.set_text(self.msg)


def addCobraIdCallback(fig, pfi, gfm=None, pfsConfig=None):
    """Add a callback to """

    onclick = ShowCobra(fig.gca(), pfi, gfm, pfsConfig)
    fig.canvas.mpl_connect('button_press_event', onclick)

    return onclick


def showDesignPhotometry(pfsConfig=None, butler=None, dataId=None, showDesign=None,
                         showGaia=True, gaiaDir=".",
                         useAB=True, band=2, maglim=None, marker='+', cmap="viridis", preserveLimits=True):
    if dataId is None:
        assert pfsConfig is not None
        if showDesign is None:
            showDesign = True
    else:
        assert butler is not None

        pfsConfig = butler.get("pfsConfig", dataId)
        md = butler.get("raw_md", dataId)

        altitude = md['ALTITUDE']
        pa = md['INST-PA']
        utc = f"{md['DATE-OBS']} {md['UT']}"

        boresight = [[pfsConfig.raBoresight], [pfsConfig.decBoresight]]

        if showDesign is None:
            showDesign = False

    pfsConfig = pfsConfig[np.logical_and(pfsConfig.spectrograph == 1,
                                         pfsConfig.targetType != TargetType.ENGINEERING)]

    muJy = 1e-3*np.array(pfsConfig.psfFlux).T[band]
    if useAB:
        brightness = 23.9 - 2.5*np.log10(muJy)
        try:
            plt.matplotlib.cm.get_cmap(cmap + "_r")
            cmap += "_r"
        except ValueError:
            pass

    vmin = int(np.nanmin(brightness)) if useAB else None
    if maglim is not None:
        vmax = maglim
    else:
        vmax = int(np.nanmax(brightness)) + 1 if useAB else None

    cmap = plt.matplotlib.cm.get_cmap(cmap)
    norm = plt.matplotlib.colors.Normalize(vmin, vmax)

    if maglim is None:
        maglim = 100
    if not showDesign and maglim < 100:
        vmax = maglim

    S = None
    if showDesign:
        for t in set(pfsConfig.targetType):
            ll = pfsConfig.targetType == t
            x, y = pfsConfig.pfiNominal.T

            ll = np.logical_and(ll, np.isfinite(x + y))

            if sum(ll) == 0:
                continue

            x, y = x[ll], y[ll]
            edgecolor = cmap(norm(brightness[ll]))  # seems to be needed to get coloured empty symbols

            _marker = {TargetType.SCIENCE: 'o', TargetType.FLUXSTD: '*'}[t]
            S = plt.scatter(x, y, c=brightness[ll], label=TargetType(t),
                            marker=_marker, edgecolor=edgecolor, cmap=cmap, norm=norm)
            S.set_facecolor("none")  # empty the symbols; facecolor='none' keeps them filled

        plt.legend(loc='upper left')

    xlim = plt.xlim()
    ylim = plt.ylim()
    if len(plt.gca().get_lines()) + len(plt.gca().collections) == 0:  # n.b. ax.has_data() doesn't work
        xlim = None, None
        ylim = xlim

    if showGaia and dataId is not None:
        plateName = pfsConfig.designName
        if plateName == 'raster_ngc6633_2':
            plateName = "ngc6633_astrometryFalse"

        if gaiaDir is None:
            gaiaDir = "."

        gaia = None
        for g in ["gaia3", "gaia"]:
            for pname in [plateName, re.sub("_.*$", "", plateName)]:
                fname = os.path.join(gaiaDir, f"{g}_{pname}.csv")
                if os.path.exists(fname):
                    gaia = pd.read_csv(fname)
                    break

            if gaia is not None:
                break

        if gaia is None:
            print(f"Unable to find GAIA catalogue for {plateName}")
            return

        if False:
            "gaia_raster11.csv"
            "gaia_ngc6828.csv"

        xmm, ymm = CoordinateTransform(np.stack((gaia.ra, gaia.dec)), mode="sky_pfi", za=90.0 - altitude,
                                       pa=pa, cent=boresight, time=utc)[0:2]
        gaiaMag = [gaia.phot_g_mean_mag, gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag][band]
        ll = gaiaMag < maglim

        if marker in ['+']:
            edgecolor = None
        else:
            edgecolor = cmap(norm(gaia.phot_rp_mean_mag[ll]))  # needed to get coloured empty symbols

        S = plt.scatter(xmm[ll], ymm[ll], c=gaia.phot_rp_mean_mag[ll], marker=marker,
                        edgecolor=edgecolor, cmap=cmap, norm=norm)
        if edgecolor is not None:
            S.set_facecolor("none")  # empty the symbols; facecolor='none' keeps them filled

        if False:
            xmm, ymm = CoordinateTransform(np.stack((pfsConfig.ra, pfsConfig.dec)), mode="sky_pfi",
                                           za=90.0 - altitude, pa=pa, cent=boresight, time=utc)[0:2]

            plt.plot(xmm, ymm, 'x', color='red')

    if S is not None:
        plt.colorbar(S, label="AB", shrink=0.8)

    if preserveLimits:
        plt.xlim(xlim)
        plt.ylim(ylim)

    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"0x{pfsConfig.pfsDesignId:x}  {pfsConfig.designName}     {pfsConfig.filterNames[1][band]}")

    plt.gca().set_aspect(1)
