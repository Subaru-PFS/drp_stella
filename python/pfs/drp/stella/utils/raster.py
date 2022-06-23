import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import astropy.units
import astropy.coordinates

import lsst.afw.image as afwImage
from pfs.datamodel import PfsDesign

from pfs.datamodel.pfsConfig import TargetType
from pfs.datamodel import PfsDesign
from pfs.utils.coordinates.CoordTransp import CoordinateTransform

__all__ = ["raDecStrToDeg", "makeDither", "makeCobraImages", "makeSkyImageFromCobras",
           "plotSkyImageOneCobra", "plotSkyImageFromCobras", "calculateOffsets", "offsetsAsQuiver",
           "getWindowedSpectralRange", "getWindowedFluxes", ]


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

if False:                                # !!
    from lsst.ip.isr import AssembleCcdTask

    config = AssembleCcdTask.ConfigClass()
    config.doTrim = True
    assembleTask = AssembleCcdTask(config=config)

    #--------------

    from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

    config = ExtractSpectraTask.ConfigClass()

    config.validate()

    extractSpectra = ExtractSpectraTask(config=config)


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
    usePfsArm:  
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
        from lsst.ip.isr import AssembleCcdTask

        config = AssembleCcdTask.ConfigClass()
        config.doTrim = True
        assembleTask = AssembleCcdTask(config=config)

        #--------------

        from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

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
            exp[amp.getRawBBox()].image.array -= np.nanmedian(exp[amp.getRawHorizontalOverscanBBox()].image.array)

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
    
    return Dither(dataId["visit"], ra, dec, fluxes, pfsConfig, md)


def makeCobraImages(dithers, side=4, pixelScale=0.025, R=50, fiberIds=None,
                    usePFImm=False, subtractBkgd=True, setUnimagedPixelsToNaN=False):
    """
    dithers: a list of (visit, ra, dec, fluxes, pfsConfig, md)
             where (ra, dec) are the boresight pointing, fluxes are the fluxes in the fibres,
             and md is the image metadata
    side: length of side of image, arcsec
    pixelScale: size of pixels, arcsec
    R: radius of fibre, microns
    fiberIds: only make images for these fibers
    usePFImm: make image in microns on PFI.  In this case, replace "arcsec" in side/pixelScale by "micron"
    subtractBkgd: subtract an estimate of the background

    Returns:
       images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
       extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)
    """

    if fiberIds is not None:
        try:
            fiberIds[0]
        except TypeError:
            fiberIds = [fiberIds]

    R_micron = R                        # R in microns, before mapping to arcsec on PFI
    size = None
    for d in dithers:
        fluxes = d.fluxes
        ra = d.ra
        dec = d.dec
        md = d.md
        pfsConfig = d.pfsConfig

        if subtractBkgd:
            bkgd = np.nanpercentile(fluxes, [5])[0]
            fluxes = fluxes - bkgd

        cosDec = np.cos(np.deg2rad(dec))
        boresight = [[pfsConfig.raBoresight], [pfsConfig.decBoresight]]

        altitude = md['ALTITUDE']
        insrot = None # md['INSROT']      # Not used with mode="sky_pfi"
        pa = md['INST-PA']
        utc = f"{md['DATE-OBS']} {md['UT']}"

        if usePFImm:
            try:
                x, y = CoordinateTransform(np.stack(([ra], [dec])), mode="sky_pfi", za=90.0 - altitude,
                                           inr=insrot, pa=pa, cent=boresight, time=utc)[0:2]
            except ValueError:          # not on sky; no valid boresight
                x, y = np.zeros_like(ra), np.zeros_like(dec)

            R = np.full(len(pfsConfig), R_micron)
        else:
            # The sky_pfi and pfi_sky transforms don't roundtrip, so we can't trivially
            # calculate the fibre radius in arcsec by transformin from/to the sky.  We could use
            # sky_pfi, then pfi_sky twice to measure the radius, but instead we can guess a value in arcsec,
            # find what it is in microns, and scale.  I.e. sky_pfi twice.

            R_asec = 0.5  # approximate radius in arcsec; the actual value doesn't matter
            x, y = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec])),
                                       mode="sky_pfi", za=90.0 - altitude,
                                       inr=insrot, pa=pa, cent=boresight, time=utc)[0:2]
            x2, y2 = CoordinateTransform(np.stack(([pfsConfig.ra], [pfsConfig.dec + R_asec/3600])),
                                         mode="sky_pfi", pa=pa, za=90.0 - altitude,
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
            weights = np.zeros_like(images)

            x0, y0 = x, y               # offset relative to first dither

        if usePFImm:
            dx, dy = 1e3*(x - x0), 1e3*(y - y0)   # microns
        else:
            dx, dy = 3600*(x - x0)*cosDec, 3600*(y - y0)

        xc, yc = hsize + dx/pixelScale, hsize + dy/pixelScale

        for i, (R, fid) in enumerate(zip(R, pfsConfig.fiberId)):
            if fiberIds and fid not in fiberIds:
                continue
            illum = np.where(np.hypot(I - yc, J - xc) < R/pixelScale, 1.0, 0.0)
            images[i, I, J] += illum*fluxes[i]
            weights[i, I, J] += illum

    images /= np.where(weights == 0, 1, weights)
    if setUnimagedPixelsToNaN:
        images[weights == 0] = np.NaN

    return images, extent


def makeSkyImageFromCobras(pfsConfig, images, pixelScale, setUnimagedPixelsToNaN=True,
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
    weights = np.zeros_like(pfiIm)
    pfiImMask = np.zeros_like(pfiIm, dtype=bool)

    nCobra = len(pfsConfig)             # sum(pfsConfig.targetType != TargetType.ENGINEERING)
    stampOffset = np.empty((nCobra, 2))  # offset of stamp from its true position due to zc = int(z + 0.5)
                                         # noqa: E114, E116     rounding to snap to grid (pass for flake8)

    for i in range(nCobra):
        xx, yy = np.array(((x[i] - xmin)*cosDec, y[i] - ymin))*scale
        if not np.isfinite(xx + yy):
            if pfsConfig.targetType[i] not in (TargetType.ENGINEERING, TargetType.UNASSIGNED):
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
            good = np.isfinite(images[i])
            pfiIm[y0:y1, x0:x1][good] += images[i][good]
            weights[y0:y1, x0:x1][good] += 1
            pfiImMask[y0:y1, x0:x1][good] = True
        except ValueError:
            print(f"fiberId {pfsConfig.fiberId[i]} doesn't fit in image")

    if setUnimagedPixelsToNaN:
        pfiIm[pfiImMask == False] = np.NaN

    pfiIm /= np.where(weights == 0, 1, weights)

    raPixsize  = (xmax - xmin)/(imsize - 1)  # noqa E221: size of pixels in ra direction
    decPixsize = (ymax - ymin)/(imsize - 1)  # :          and in dec direction
    extent = (xmin - raPixsize/2, xmax + raPixsize/2, ymin - decPixsize/2, ymax + decPixsize/2)

    return pfiIm, extent, stampOffset


def plotSkyImageOneCobra(fiberId, pfsConfig, images, extent_CI, usePFImm=False, vmin=None, vmax=None):
    """Plot the synthesised image of the sky from dithered spectra
    fiberId: desired cobra
    pfsConfig: PFI configuration
    images: sky image for each cobra in pfsConfig,  ndarray `(len(pfsConfig, n, n)`
    extent_CI: (x0, x1, y0, y1) giving the corners of images[n] (as passed to plt.imshow)

    images, extent_CI are as returned by makeCobraImages
    """
    cmap = plt.matplotlib.cm.get_cmap().copy()
    cmap.set_bad(color='black')

    ims = plt.imshow(images[pfsConfig.selectFiber(fiberId)], origin='lower', interpolation='none',
               extent=extent_CI, vmin=vmin, vmax=vmax)
    plt.gca().set_aspect(1)

    plt.plot(0, 0, '+', color='red')

    if usePFImm:
        plt.xlabel(r"$\Delta x$ ($\mu m$)")
        plt.ylabel(r"$\Delta y$ ($\mu m$)")
    else:
        plt.xlabel(r"$\Delta\alpha$ (arcsec)")
        plt.ylabel(r"$\Delta\delta$ (arcsec)")

    plt.title(f"fiberId = {fiberId}")

    return ims


def plotSkyImageFromCobras(pfsConfig, pfiIm, extent, stampOffset, usePFImm=False, showNominalPosition=True,
                           vmin=None, vmax=None):
    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])  # == 1/cosDec

    if usePFImm:
        x, y = pfsConfig.pfiNominal.T
    else:
        x, y = pfsConfig.ra, pfsConfig.dec

    cmap = plt.matplotlib.cm.get_cmap().copy()
    cmap.set_bad(color='black')

    ims = plt.imshow(pfiIm, origin='lower', interpolation='none', cmap=cmap,
                     extent=extent, vmin=vmin, vmax=vmax)
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
    xoff, yoff = np.average(x, axis=(1, 2), weights=weights), np.average(y, axis=(1, 2), weights=weights)

    return xoff, yoff


def offsetsAsQuiver(pfsConfig, xoff, yoff, usePFImm=False, quiverLen=None, quiverLabel=None):
    """Plot the offsets (as calculated by calculateOffsets) as a quiver
    """
    if usePFImm:
        cosDec = 1
        x, y = pfsConfig.pfiNominal.T
        if quiverLen is None:
            quiverLen = 50
        if quiverLabel is None:
            quiverLabel = "mm"
    else:
        cosDec = np.cos(np.deg2rad(pfsConfig.decBoresight))
        x, y = pfsConfig.ra, pfsConfig.dec
        if quiverLen is None:
            quiverLen = 0.5
        if quiverLabel is None:
            quiverLabel = "arcsec"
    xoff -= 0.1; yoff -= 0.1
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


class ShowCobra:
    """Show a cobraId and possibly fiberId on right-click"""

    def __init__(self, ax, pfi, gfm=None):
        self.ax = ax
        self.pfi = pfi
        self.gfm = gfm
        self.text = ax.text(0, 0, "", va="bottom", ha="left", color='white', transform=ax.transAxes)
        self.circle = None
        #
        # Save the values
        #
        self.cobraId = -1
        self.fiberId = -1
        #
        # For debugging
        #
        self.event = None
        self.exception = None
        self.message = ""

    def __call__(self, event):
        self.event = event
        if self.circle:
            self.circle.remove()
            self.circle = None

        self.msg = ""
        try:
            if event.button in [3]:
                x, y = event.xdata, event.ydata
                self.cobraId = np.argmin(np.abs(self.pfi.centers - (x + 1j*y))) + 1
                self.msg += f"cobraId {self.cobraId:4}"

                if self.gfm:
                    self.fiberId = self.gfm.fiberId[self.gfm.cobraId == self.cobraId][0]
                    self.msg += f"  fiberId {self.fiberId:4}"

                self.circle = Circle((x, y), 2, facecolor=None, edgecolor='red', fill=False, alpha=1)
                self.ax.add_patch(self.circle)
                    
        except Exception as e:
            self.exception = e
        else:
            self.exception = None

        self.text.set_text(self.msg)

def addCobraIdCallback(fig, pfi, gfm=None):
    """Add a callback to """

    onclick = ShowCobra(fig.gca(), pfi, gfm)
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    return onclick
