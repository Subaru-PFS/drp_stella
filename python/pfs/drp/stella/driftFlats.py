import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import lsst.afw.image as afwImage
from pfs.datamodel.pfsConfig import TargetType
from pfs.drp.stella import DetectorMap, FiberProfile, FiberProfileSet, SpectrumSet
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundTask


class DriftSet:
    """Handle a set of data used to generate a drift flat"""
    
    def __init__(self, name, home_visit, drift_visits, dxs=[(-150, 150)], ttype=TargetType.ENGINEERING):
        self.name = name
        self.home_visit = home_visit
        self.drift_visits = drift_visits
        self.dxs = dxs
        self.ttype = ttype
        self.xcs = None
        self.fiberId = None
        self.imageWidth = None
        # analysis arrays, indexed by the index into drift_visits
        self.lefts = {}    # sets of left- and right-hand boundaries for each visit in self.drift_visits
        self.rights = {}
        self.driftedModels = {}
        self.flats = {}
        self.corrections = {}
        self.profiles = {}

    def __len__(self):
        return len(self.drift_visits)

#
# E.g. 
# driftSets = [DriftSet("ALF", 106654, [106638], [(-150, 150)]),
#              DriftSet("Yuki-2024-03-10:2", 107396, [107417, 107418], [(-165, 35), (-35, 165)]),
#             ]
#
# N.b. E.g. 
# reduceExposure.py /work/drp --calib=/work/drp/CALIB --rerun=rhl/tmp \
#       --id visit=107394..107396 \
#        --config targetType=[ENGINEERING] isr.doDark=True isr.doIPC=False repair.doCosmicRay=False \
#                 doBoxcarExtraction=True boxcarWidth=11 doMeasureLines=False doAdjustDetectorMap=False \
#                 useCalexp=True

def driver(butler, driftSet, arm, spectrograph, step=0.5, doPlot=False):
    dataId = dict(visit=driftSet.home_visit, arm=arm, spectrograph=spectrograph)

    home = butler.get("calexp", dataId)
    homeDetMap = butler.get("detectorMap_used", dataId)
    pfsConfig = butler.get("pfsConfig", dataId).select(spectrograph=dataId["spectrograph"],
                                                       targetType=driftSet.ttype)
    spec = butler.get("pfsArm", dataId)

    print("buildFiberProfiles")
    fiberProfiles, spectra = buildFiberProfiles(home, homeDetMap, pfsConfig, spec)
    scatteredLight = estimateScatteredLight(home, homeDetMap, fiberProfiles, spectra)

    for vset in range(len(driftSet)):
        dataId["visit"] = driftSet.drift_visits[vset]
        exp = butler.get("calexp", dataId)
        detMap = butler.get("detectorMap_used", dataId)

        exp.image -= scatteredLight

        print(f"\t{vset} {dataId['visit']}  findSlitMotion")
        findSlitMotion(driftSet, vset, exp, detMap, pfsConfig, fiberProfiles,
                       doPlot=doPlot, title=f"{dataId}")

        print(f"\t{vset} {dataId['visit']}  estimateTraceIllumination")
        estimateTraceIllumination(driftSet, vset, exp, rowwidth=300)

        print(f"\t{vset} {dataId['visit']}  calculateModel")
        calculateModel(driftSet, vset, home, homeDetMap, fiberProfiles, spectra, pfsConfig.fiberId, step,
                       butler=butler, dataId=dataId)
        #
        # Add the scattered light to the model
        #
        driftedModel = driftSet.driftedModels[vset]
        flux_drift = np.nansum(driftedModel.array)
        flux_home = np.nansum(home.image.array)

        driftedModel.array += scatteredLight.array*flux_drift/flux_home

        print(f"\t{vset} {dataId['visit']}  calculateOneDriftFlat")
        calculateOneDriftFlat(driftSet, vset, exp)


def buildFiberProfiles(calexp, detMap, pfsConfig, spec):
    spec.flux[(spec.mask & ~spec.flags["REFLINE"]) != 0x0] = np.NaN
    spectra = SpectrumSet.fromPfsArm(spec)
    #
    # Build the profiles
    #
    config = BuildFiberProfilesTask.ConfigClass()
    config.doBlindFind = False
    config.profileRadius = 11
    buildFiberProfilesTask = BuildFiberProfilesTask(config)

    fiberProfiles = buildFiberProfilesTask.run(calexp, None, detectorMap=detMap, pfsConfig=pfsConfig).profiles

    return fiberProfiles, spectra


def estimateScatteredLight(calexp, detMap, fiberProfiles, spectra):
    """Estimate the scattered light"""
    config = SubtractBackgroundTask.ConfigClass()
    config.binSize = 128
    config.statisticsProperty = "MEDIAN"

    subtractBackgroundTask = SubtractBackgroundTask(config)

    fts = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)

    calexp = calexp.clone()
    calexp.image -= spectra.makeImage(calexp.getDimensions(), fts)

    bkgd = subtractBackgroundTask.run(calexp).background[0][0]
    scatteredLight = bkgd.getImageF()

    return scatteredLight


class Profile:
    # A class used to perform linear interpolation in an array, designed to be passed to brentq
    def __init__(self, array, c=0):
        self._array = array
        self._n = len(array)
        self.c = c

    def __call__(self, x):
        """Linear interpolation"""
        ix = np.floor(x).astype(int)
        if False:
            if ix < 0:
                ix = 0
            elif ix >= self._n - 1:
                ix = self._n - 2

        val = ((ix + 1) - x)*self._array[ix] + (x - ix)*self._array[ix + 1]
        return val - self.c

    def __len__(self):
        return self._n


def findSlitMotion(driftSet, vset, exp, detMap, pfsConfig, fiberProfiles, rowwidth=200,
                   doPlot=False, title=""):
    """Find the motion of the slit

    Supposed to be linear, but not well synchronised with the lamp turning on.
    We'll correct for non-linear motion later

    rowwidth:   number of rows to average
    doPlot:
    title:
    """

    #
    # Find the limits of the drift by analysing the drifted exposure.
    # We assume that most/many of the fibres don't overlap
    #
    dx = driftSet.dxs[vset]
    width = dx[1] - dx[0] + 1   # guess!
    driftSet.imageWidth = detMap.getBBox().getWidth()

    profile = np.median(exp.image.array[2000-rowwidth//2:2000+rowwidth//2+1], axis=0)

    prof = Profile(profile)

    if doPlot:
        plt.plot(profile, color="black")
        plt.title(title)

    left, right = {}, {}
    driftSet.lefts[vset], driftSet.rights[vset] = left, right

    for i, fid in enumerate(fiberProfiles.fiberId):
        left[fid]  = np.NaN
        right[fid] = np.NaN

        fhole = pfsConfig.select(fiberId=fid).fiberHole[0]
        if fhole in [308, 316, 336, 341, 360]:   # too close to the centre of the detector pair in b/r/m
            continue

        xc = detMap.getXCenter(fid, 2000)
        ixc = int(xc + 0.5)

        lev = np.mean(profile[ixc - int(0.25*width):ixc + int(0.25*width)+1])

        # Brent [a, b] for finding edges
        al, bl = max([0, xc + 1.05*dx[0]]), xc                                        # left side
        ar, br = xc                       , min([xc + 1.05*dx[1], exp.getWidth()-1])  # right side
        prof.c = 0.9*lev

        if doPlot:
            plt.axvline(xc, color=f"C{i}", label=f"{fid}", alpha=0.25)

            ixc = int(xc + 0.5)
            lev = np.mean(profile[ixc - int(0.25*width):ixc + int(0.25*width)+1])

            plt.plot([al, br], [prof.c, prof.c], color=f"C{i}")

        try:
            left[fid]  = brentq(prof, al, bl) if al >= 0 else np.NaN            
            right[fid] = brentq(prof, ar, br) if br < len(prof) - 1 else np.NaN
        except ValueError as e:
            print(f"Failed for fiberId {fid}  (fiberHole {fhole}): {e}")
            continue

        if doPlot:
            plt.axvline(left[fid], color=f"C{i}", alpha=1, zorder=-1)
            plt.axvline(right[fid], color=f"C{i}", alpha=1, zorder=-1)

    # Handle missing limits
    dleft = []
    dright = []
    xc = []
    fids = []

    for i, fid in enumerate(sorted(list(set(left) & set(right)))):
        fids.append(fid)
        xc.append(detMap.getXCenter(fid, 2000))
        dleft.append(xc[i] - left[fid])
        dright.append(right[fid] - xc[i])

    dleftbar = np.nanmedian(dleft)
    drightbar = np.nanmedian(dright)

    for i in range(len(fids)):
        if np.isnan(dleft[i]):
            dleft[i] = dleftbar
            left[fids[i]] = xc[i] - dleft[i]
        if np.isnan(dright[i]):
            dright[i] = drightbar
            right[fids[i]] = xc[i] + dright[i]

    driftSet.xcs = xc
    driftSet.fiberId = fids

    driftSet.profiles[vset] = profile


def showTraceEdges(pfsConfig, driftSet, vset, doPlot=True, doPrint=True, title=""):
    fids = driftSet.fiberId
    left = driftSet.lefts[vset]
    right = driftSet.rights[vset]
    profile = driftSet.profiles[vset]
    xc = driftSet.xcs

    dx = driftSet.dxs[vset]
    width = dx[1] - dx[0] + 1   # guess!

    if doPlot:
        plt.plot(profile, color="black")
        plt.title(title)

    if doPrint:
        print(f"{'fid':4s} {'hole':3s} {'xc':7s}  "
              f"{'left':7s} {'right':7s}  {'width':5s}  {'dleft':6s} {'dright':6s}")

    for i, fid in enumerate(fids):
        fhole = pfsConfig.select(fiberId=fid).fiberHole[0]

        if doPrint:
            print(f"{fid:4d} {fhole:3d}  {xc[i]:7.2f}  "
                  f"{left[fid]:7.2f} {right[fid]:7.2f}  "
                  f"{right[fid] - left[fid]:6.2f} "
                  f"{xc[i] - left[fid]:6.2f} {right[fid] - xc[i]:6.2f}")

        if doPlot:
            plt.axvline(xc[i], color=f"C{i}", label=f"{fid}", alpha=0.25)

            ixc = int(xc[i] + 0.5)
            lev = np.mean(profile[ixc - int(0.25*width):ixc + int(0.25*width)+1])

            al = max([0, xc[i] + 1.05*dx[0]])
            br = min([xc[i] + 1.05*dx[1], driftSet.imageWidth - 1])

            c = 0.9*lev
            plt.plot([al, br], [c, c], color=f"C{i}")

            plt.axvline(left[fid], color=f"C{i}", alpha=1, zorder=-1)
            plt.axvline(right[fid], color=f"C{i}", alpha=1, zorder=-1)

    if doPlot:
        plt.legend(ncol=2)


def estimateTraceIllumination(driftSet, vset, exp, rowwidth=300):
    """Estimate the correction to the effective illumination of each column

    We do this based on averaging the bands produced by the different fibres
    """
    profile = np.median(exp.image.array[2000-rowwidth//2:2000+rowwidth//2+1], axis=0)

    fids = driftSet.fiberId
    left = driftSet.lefts[vset]
    right = driftSet.rights[vset]

    width = []
    for fid in fids:
        r0 = left[fid]
        r1 = right[fid]

        if np.isnan(r0 + r1):
            continue

        width.append(int(r1 - r0 + 1))
        stripWidth = np.max(width)

    correctionArr = np.full((len(fids), stripWidth), np.NaN)
    for i, fid in enumerate(fids):
        #xc = detMap.getXCenter(fid, 2000)

        r0 = left[fid]
        r1 = r0 + stripWidth

        if r0 >= 0:
            j0 = 0
            r0 = int(r0)
        else:
            j0 = int(-r0) + 1
            r0 = 0

        width = exp.getWidth()
        if r1 >= width:
            j1 = stripWidth - (int(r1) - width)
            r1 = width
        else:
            j1 = stripWidth
            r1 = int(r1)

        pp = profile[r0:r1].copy()
        pp /= np.median(pp)
        correctionArr[i, j0:j1] = pp

    correction = np.nanmedian(correctionArr, axis=0)
    driftSet.corrections[vset] = correction


def calculateModel(driftSet, vset, calexp, detMap, fiberProfiles, spectra, fiberId, step=0.5,
                   butler=None, dataId=None):
    """
    Calculate the model of the drifted fibres. This is the expensive step


    Note that we don't (now) assume that they move at a constant rate, but interpret the `correction` vector,
    derived from the column-to-column variation in brightness of the drift signal from each fibre,
    as a measure of the position of the slit; this is used to calculate `dxs`.  

    It may be that in fact the light source is fluctuating, but this seems rather less likely, both a priori
    and because the pattern seems to be stable from run to run (check me!) and depend on the direction of
    the motion

    calexp: home_visit Exposure
    step:  pixels
"""
    fids = driftSet.fiberId
    correction = driftSet.corrections[vset]
    left = driftSet.lefts[vset]
    right = driftSet.rights[vset]
    xc = driftSet.xcs

    ccorr = np.cumsum(2 - correction)  # i.e. flip the sign of (correction - 1)
    ccorr -= ccorr[0]

    # Find the pixel range over which the spectra moved (roughly the motion of the slit,
    # but after optical distortion)
    offset0, offset1 = [], []
    for i, fid in enumerate(fids):
        offset0.append(left[fid] - xc[i])
        offset1.append(right[fid] - xc[i])

    offset0 = np.median(offset0)
    offset1 = np.median(offset1)

    dxs = offset0 + ccorr*(offset1 - offset0)/ccorr[-1]

    if step != 1:
        i_in = np.arange(0, len(dxs))
        i_out = np.arange(0, len(dxs), step)
        dxs = np.interp(i_out, i_in, dxs)

    spatialOffsets0 = detMap.getSpatialOffsets()
    spectralOffsets0 = detMap.getSpectralOffsets()

    driftedModel = None
    for i, dx in enumerate(dxs):
        print(f"\t\tdx: {dx:.1f} {int(100*i/(len(dxs) - 1))}%", end='\r', flush=True)
        if butler is None:              # reset the DetectorMap; doesn't work (nor does detMap.clone())
            detMap.setSlitOffsets(spatialOffsets0, spectralOffsets0)
        else:
            assert dataId is not None
            detMap = butler.get("detectorMap_used", dataId)

        bad = []
        for fid in fiberId:
            if False and fid in [1303, 1953]:
                continue

            if True:  # workaround inability to offset part of fibre off the chip
                xCenter = detMap.getXCenter(fid) + dx
                if np.min(xCenter) < 0 or np.max(xCenter) >= detMap.bbox.endX - 1:
                    bad.append(fid)
                    if False and fid != 1953:
                        print("RHL", bad)
                        import pdb; pdb.set_trace() 
                else:
                    detMap.setSlitOffsets(fid, detMap.getSpatialOffset(fid) + dx,
                                          detMap.getSpectralOffset(fid))
            else:
                detMap.setSlitOffsets(fid, detMap.getSpatialOffset(fid) + dx,
                                      detMap.getSpectralOffset(fid))

        try:
            fts = fiberProfiles.makeFiberTracesFromDetectorMap(detMap, ignoreFiberId=bad)
        except Exception as e:
            print(dx, e)
            continue

        image = spectra.makeImage(calexp.getDimensions(), fts)
        if driftedModel is None:
            driftedModel = image
            driftSet.driftedModels[vset] = driftedModel
        else:
            driftedModel += image

    print("")

    driftedModel /= len(dxs)
    # we don't want to divide by the background model, so set missing pixels to NaN before we add it
    driftedModel.array[driftedModel.array == 0] = np.NaN


def calculateOneDriftFlat(driftSet, vset, exp):
    """Calculate the flat from a single drift exposure
    """

    flat = exp.clone()
    driftSet.flats[vset] = flat

    driftedModel = driftSet.driftedModels[vset]

    flat.image /= driftedModel
    flat.image /= np.nanmedian(flat.image.array[:, np.nanmedian(driftedModel.array, axis=0) > 10])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def mergeOneDriftFlats(driftSet, margin=3):
    ncol = driftSet.imageWidth
    nflat = len(driftSet.drift_visits)
    overlaps = np.zeros((nflat, ncol), dtype=int)
    fids = np.zeros_like(overlaps)

    for vs in range(nflat):
        for i, (fid, l, r) in enumerate(zip(driftSet.lefts[vs],
                                            driftSet.lefts[vs].values(), driftSet.rights[vs].values()), 1):
            il, ir = int(l), int(r)
            il = min(il + margin, ncol - 1)
            ir = max(0, ir - margin)

            overlaps[vs, il:ir+1] = 2**vs
            fids[vs, il:ir+1] = fid

            if False and vs == 0:
                print(fid, il, ir)

    overlaps = np.sum(overlaps, axis=0)

    # ends of ranges of overlapping columns ("blocks") from the drift_visits
    endblock = np.arange(1, ncol)[np.diff(overlaps) != 0]
    endblock = np.append(endblock, ncol)

    nblock = len(endblock)

    # the median values in each vset for the range of columns in a block
    block_data = np.empty((nflat, nblock))
    block_fids = np.empty((nflat, nblock), dtype=int)
    start_end = np.empty((2, nblock), dtype=int)

    start = 0
    for i, end in enumerate(endblock):
        for vs in range(nflat):
            im = driftSet.flats[vs].image.array[1000:3000, start:end]
            block_data[vs, i] = np.nanmedian(im)

            fid = set(fids[vs][start:end])
            if len(fid) != 1:
                if set(fid).difference({1618, 1638}):
                    print(fid)

            block_fids[vs, i] = list(fid)[0]
            start_end[:, i] = (start, end)

        start = end

    if False:
        for i, end in enumerate(endblock):
            j = i//nflat
            if len([_ for _ in block_fids[:, i] if _ > 0]) > 1:
                start, end = start_end[:, i]
                print(f"{j:3d} {start:4d}:{end:4d}  {block_fids[:, i]} {block_data[:, i]}")

    flats = driftSet.flats.copy()
    for i in range(nflat):
        flats[i] = driftSet.flats[i].clone()

    assert nflat == 2   # I haven't thought the general case through (yet?)

    for i in range(1, nblock, 2):
        j = i//2

        if j%2 == 0:
            vset = 1
            factor = block_data[0, i]/block_data[1, i]
        else:
            vset = 0
            factor = block_data[1, i]/block_data[0, i]

        # Update the scaling for the flat from visits vset
        for di in range(3):
            if i + di < nblock:
                block_data[vset, i + di] *= factor

        i1 = i + 1 if i + 1 < nblock else i
        i2 = i + 2 if i + 2 < nblock else i
        fid = set([block_fids[:, i][vset], block_fids[:, i1][vset], block_fids[:, i2][vset]])
        # handle the centre where multiple fibres overlap.  Or really, avoid the question
        if fid.difference({1618, 1638}):
            if 0 in fid:
                continue
            #assert len(fid) == 1, f"{fid} {block_fids[:, i][vset]}"
        else:
            fid = [block_fids[:, i][vset]]

        fid = list(fid)[0]
        assert fid == block_fids[:, i][vset], f"{j}: {fid} == {block_fids[:, i]}[0]"

        start, end = int(driftSet.lefts[vset][fid]), int(driftSet.rights[vset][fid])
        flats[vset].image.array[:, start:end] *= factor

    #  avoid extra pixels in the region near the left/right edges where the profile's rolling off
    extra = 15
    margin += extra                     # margin eats into the "flat topped" region
    margin = 3 + extra
    for vset in range(nflat):
        for l, r in zip(driftSet.lefts[vset].values(), driftSet.rights[vset].values()):
            il = max([int(l) - extra, 0])
            ir = min([int(r) + extra, driftSet.imageWidth - 1])

            flats[vset].image.array[:, il:il + margin] = np.NaN
            flats[vset].image.array[:, ir - margin:ir] = np.NaN

    flat = np.nanmedian(np.array([_.image.array for _ in flats.values()]), axis=0).astype(np.float32)

    return afwImage.ImageF(flat), flats, start_end, endblock, block_data


