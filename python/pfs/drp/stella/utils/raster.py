import numpy as np
import matplotlib.pyplot as plt

import astropy.units
import astropy.coordinates

from pfs.datamodel.pfsConfig import TargetType
from pfs.utils.coordinates.CoordTransp import CoordinateTransform

__all__ = ["raDecStrToDeg", "makeDither", "makeCobraImages", "makeSkyImageFromCobras",
           "plotSkyImageOneCobra", "plotSkyImageFromCobras", "calculateOffsets", "offsetsAsQuiver",
           "getWindowedSpectralRange", ]


def raDecStrToDeg(ra, dec):
    """Convert ra, dec as hexadecimal strings to degrees"""

    return (astropy.coordinates.Angle(ra, unit=astropy.units.h).degree,
            astropy.coordinates.Angle(dec, unit=astropy.units.deg).degree)


class Dither:
    """Information associated with a dithered PFS exposure"""
    def __init__(self, visit, ra, dec, fluxes, pfsConfig, md):
        self.visit = visit
        self.ra = ra
        self.dec = dec
        self.fluxes = fluxes
        self.pfsConfig = pfsConfig
        self.md = md


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


def makeDither(butler, dataId, lmin=665, lmax=700, targetType=None, dradec=None, pfsDesignId=None):
    """Return an element of the dithers[] array that's passed to makeCobraImages"""
    if pfsDesignId is None:             # use real pfsConfig
        pfsConfig = butler.get("pfsConfig", dataId)
    else:
        from pfs.datamodel.pfsConfig import PfsDesign
        if dradec is None:              # check that we're faking data
            raise RuntimeError("You must set dradec if you want to specify a pfsDesignId")

        pfsConfig = PfsDesign.read(pfsDesignId, "/data/pfsDesign")

    pfsConfig = pfsConfig[pfsConfig.spectrograph == dataId["spectrograph"]]
    select = np.ones(len(pfsConfig), dtype=bool)  # only include these fibres
    if targetType is None:
        select = np.logical_not(np.logical_or(pfsConfig.targetType == TargetType.UNASSIGNED,
                                              pfsConfig.targetType == TargetType.ENGINEERING))
    else:
        select = pfsConfig.targetType == targetType
    pfsConfig = pfsConfig[select]

    spec = butler.get("pfsArm", dataId)[select]

    md = butler.get("raw_md", dataId)
    ra, dec = raDecStrToDeg(md['RA_CMD'], md['DEC_CMD'])

    if dradec is not None:
        # Fake the ra/dec of the data for testing purposes

        pfsConfig.raBoresight, pfsConfig.decBoresight = ra, dec
        pfsConfig.dec += pfsConfig.decBoresight - np.nanmean(pfsConfig.dec)

        pfsConfig.ra -= np.nanmean(pfsConfig.ra)
        pfsConfig.ra /= np.cos(np.deg2rad(np.nanmean(pfsConfig.dec)))
        pfsConfig.ra += pfsConfig.raBoresight - np.nanmean(pfsConfig.ra)

        v = dataId["visit"]
        dra, ddec = dradec.get(v, (0, 0))
        ra  += (dra/3600)/np.cos(np.deg2rad(dec))  # noqa: E221
        dec += (ddec/3600)

    fluxes = np.nanmedian(np.where(np.logical_and(spec.wavelength >= lmin, spec.wavelength <= lmax),
                                   spec.flux, np.NaN), axis=1)

    return Dither(dataId["visit"], ra, dec, fluxes, pfsConfig, md)


def makeCobraImages(dithers, side=4, pixelScale=0.025, R=50, usePFImm=False):
    """
    dithers: a list of (visit, ra, dec, fluxes, pfsConfig, md)
             where (ra, dec) are the boresight pointing, fluxes are the fluxes in the fibres,
             and md is the image metadata
    side: length of side of image, arcsec
    pixelScale: size of pixels, arcsec
    R: radius of fibre, microns
    usePFImm: make image in microns on PFI.  In this case, replace "arcsec" in side/pixelScale by "micron"

    Returns:
       images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
       extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)
    """

    R_micron = R                        # R in microns, before mapping to arcsec on PFI
    size = None
    for d in dithers:
        fluxes = d.fluxes
        ra = d.ra
        dec = d.dec
        md = d.md
        pfsConfig = d.pfsConfig

        cosDec = np.cos(np.deg2rad(dec))
        boresight = [[pfsConfig.raBoresight], [pfsConfig.decBoresight]]

        altitude = md['ALTITUDE']
        insrot = md['INSROT']
        utc = f"{md['DATE-OBS']} {md['UT']}"

        if usePFImm:
            x, y = CoordinateTransform(np.stack(([ra], [dec])), "sky_pfi", 90.0 - altitude,
                                       inr=insrot, cent=boresight, time=utc)[0:2]
            x, y = x[0], y[0]
            R = np.full(len(pfsConfig), R_micron)
        else:
            # The sky_pfi and pfi_sky transforms don't roundtrip, so we can't trivially
            # calculate the fibre radius in arcsec by transformin from/to the sky.  We could use
            # sky_pfi, then pfi_sky twice to measure the radius, but instead we can guess a value in arcsec,
            # find what it is in microns, and scale.  I.e. sky_pfi twice.

            R_asec = 0.5  # approximate radius in arcsec; the actual value doesn't matter
            x, y = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec])),
                                       "sky_pfi", 90.0 - altitude,
                                       inr=insrot, cent=boresight, time=utc)[0:2]
            x2, y2 = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec + R_asec/3600])),
                                         "sky_pfi", 90.0 - altitude,
                                         inr=insrot, cent=boresight, time=utc)[0:2]
            R = np.hypot(x - x2, y - y2)*1e3  # microns corresponding to R_asec
            R = R_asec*R_micron/R   # asec

            x, y = ra, dec

        if size is None:
            size = int(side/pixelScale + 1)
            if size%2 == 0:
                size += 1
            hsize = size//2

            extent = np.array((-hsize - 0.5, hsize + 0.5, -hsize - 0.5, hsize + 0.5))*pixelScale

            I, J = np.meshgrid(np.arange(0, size), np.arange(0, size))

            nCobra = len(pfsConfig)
            images = np.zeros((nCobra, size, size))

            x0, y0 = x, y

        if usePFImm:
            dx, dy = 1e3*(x - x0), 1e3*(y - y0)   # microns
        else:
            dx, dy = 3600*(x - x0)*cosDec, 3600*(y - y0)

        xc, yc = hsize + dx/pixelScale, hsize + dy/pixelScale

        for i, (R, fid) in enumerate(zip(R, pfsConfig.fiberId)):
            images[i, I, J] += np.where(np.hypot(I - yc, J - xc) < R/pixelScale, 1.0, 0.0)*fluxes[i]

    return images, extent


def makeSkyImageFromCobras(pfsConfig, images, pixelScale, maskUnimagedPixels=True,
                           usePFImm=False, figure=None):
    """
    usePFImm: make image in microns on PFI.  In this case, replace "arcsec" in side/pixelScale by "micron"
    """
    if usePFImm:
        cosDec = 1.0
        x, y = pfsConfig.pfiNominal.T
    else:
        cosDec = np.cos(np.deg2rad(np.nanmean(pfsConfig.dec)))
        x, y = pfsConfig.ra, pfsConfig.dec

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    stampsize = images[0].shape[0]  # stamps of each cobra's images are stampsize x stampsize

    if usePFImm:
        scale = 1e3/pixelScale  # pixel/mm
    else:
        scale = 3600/pixelScale  # pixel/deg

    border = 0.55*stampsize/scale  # needed extra width (on the PFI image)

    xmin -= border/cosDec
    xmax += border/cosDec
    ymin -= border
    ymax += border

    imsize = int(max((xmax - xmin)*cosDec, ymax - ymin)*scale)
    xmax = xmin + (imsize - 1)/cosDec/scale
    ymax = ymin + (imsize - 1)/scale

    pfiIm = np.full((imsize, imsize), 0.0)
    pfiImMask = np.zeros_like(pfiIm, dtype=bool)

    nCobra = len(pfsConfig)
    stampOffset = np.empty((nCobra, 2))  # offset of stamp from its true position due to zc = int(z + 0.5)
                                         # noqa: E114, E116     rounding to snap to grid (pass for flake8)

    for i in range(nCobra):
        xx, yy = np.array(((x[i] - xmin)*cosDec, y[i] - ymin))*scale
        if not np.isfinite(xx + yy):
            print(f"pfsConfig 0x{pfsConfig.pfsDesignId:x}"
                  f" coordinates for fiberId {pfsConfig.fiberId[i]} are NaN: ({xx}, {yy})."
                  f" ObjectType is {str(TargetType(pfsConfig.targetType[i]))};"
                  )
            continue
        xc = int(xx + 0.5)
        yc = int(yy + 0.5)

        stampOffset[i] = (xc/(cosDec*scale) + xmin - x[i], yc/scale + ymin - y[i])

        x0, x1 = xc - stampsize//2, xc + stampsize//2 + 1
        y0, y1 = yc - stampsize//2, yc + stampsize//2 + 1
        try:
            pfiIm[y0:y1, x0:x1] += images[i]
            pfiImMask[y0:y1, x0:x1] = True
        except ValueError:
            print(f"Fiducial ID {pfsConfig.fiberId[i]} doesn't fit in image")

    if maskUnimagedPixels:
        pfiIm[pfiImMask is False] = np.NaN

    raPixsize  = (xmax - xmin)/(imsize - 1)  # noqa E221: size of pixels in ra direction
    decPixsize = (ymax - ymin)/(imsize - 1)  # :          and in dec direction
    extent = (xmin - raPixsize/2, xmax + raPixsize/2, ymin - decPixsize/2, ymax + decPixsize/2)

    return pfiIm, extent, stampOffset


def plotSkyImageOneCobra(fiberId, pfsConfig, images, extent_CI, usePFImm=False):
    """Plot the synthesised image of the sky from dithered spectra
    fiberId: desired cobra
    pfsConfig: PFI configuration
    images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
    extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)

    images, extent_CI are as returned by makeCobraImages
    """

    plt.imshow(images[pfsConfig.selectFiber(fiberId)], origin='lower', interpolation='none', extent=extent_CI)
    plt.gca().set_aspect(1)

    plt.plot(0, 0, '+', color='red')

    if usePFImm:
        plt.xlabel(r"$\Delta x$ ($\mu m$)")
        plt.ylabel(r"$\Delta y$ ($\mu m$)")
    else:
        plt.xlabel(r"$\Delta\alpha$ (arcsec)")
        plt.ylabel(r"$\Delta\delta$ (arcsec)")

    plt.title(f"fiberId = {fiberId}")


def plotSkyImageFromCobras(pfsConfig, pfiIm, extent, stampOffset, usePFImm=False, showNominalPosition=True):
    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])  # == 1/cosDec

    if usePFImm:
        x, y = pfsConfig.pfiNominal.T
    else:
        x, y = pfsConfig.ra, pfsConfig.dec

    plt.imshow(pfiIm, origin='lower', interpolation='none', extent=extent)
    if showNominalPosition:
        plt.plot(x + stampOffset[:, 0], y + stampOffset[:, 1], '+', alpha=0.5, color='red')

    plt.gca().set_aspect(aspect)

    if usePFImm:
        plt.xlabel(r"x (mm)")
        plt.ylabel(r"y (mm)")
    else:
        plt.xlabel(r"$\alpha$ (deg)")
        plt.ylabel(r"$\delta$ (deg)")


def calculateOffsets(images, extent_CI):
    """Calculate and return the offset of the centroid for each of the images
    images:  ndarray of shape (ncobra, xsize, ysize)
    extent_CI: tuple (x0, x1, y0, y1) giving the corners of the images[n]

    return xoff, yoff in the units specified by extent_CI
    """
    x0, x1, y0, y1 = extent_CI
    xs, ys = images[0].shape
    x, y = np.meshgrid(np.linspace(y0, y1, xs), np.linspace(x0, x1, ys))
    #
    # vectorize the calculation, at the expense of memory.  Is it worth it???
    #
    x = np.concatenate(len(images)*[x]).reshape((len(images), xs, ys))
    y = np.concatenate(len(images)*[y]).reshape((len(images), xs, ys))

    xoff, yoff = np.average(x, axis=(1, 2), weights=images), np.average(y, axis=(1, 2), weights=images)

    return xoff, yoff


def offsetsAsQuiver(pfsConfig, xoff, yoff, usePFImm=False, quiverLen=None):
    """Plot the offsets (as calculated by calculateOffsets) as a quiver
    """
    if usePFImm:
        cosDec = 1
        x, y = pfsConfig.pfiNominal.T
        if quiverLen is None:
            quiverLen = 50
            quiverLabel = "micron"
    else:
        cosDec = np.cos(np.deg2rad(pfsConfig.decBoresight))
        x, y = pfsConfig.ra, pfsConfig.dec
        if quiverLen is None:
            quiverLen = 0.5
            quiverLabel = "arcsec"
    Q = plt.quiver(x, y, xoff, yoff)

    plt.quiverkey(Q, 0.1, 0.9, quiverLen, f"{quiverLen}{quiverLabel}", coordinates='axes',
                  color='red', labelcolor='red')

    plt.gca().set_aspect(1/cosDec)

    if usePFImm:
        plt.xlabel(r"x (mm)")
        plt.ylabel(r"y (mm)")
    else:
        plt.xlabel(r"$\alpha$ (deg)")
        plt.ylabel(r"$\delta$ (deg)")
