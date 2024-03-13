import warnings
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import lsst.afw.image as afwImage
from pfs.datamodel.pfsConfig import TargetType
from pfs.drp.stella import SpectrumSet
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundTask


class DriftSet:
    """Handle a set of data used to generate a drift flat"""

    def __init__(self, name, home_visit, drift_visits,
                 dxs=[(-170, 40), (-40, 170)],
                 dataIds=None,
                 ttype=TargetType.ENGINEERING):
        """
        dataIds: the arm, spectrograph part of dataIds to process, e.g. ["b1", "r2", "n2", "r3"]
        """
        self.name = name
        self.home_visit = home_visit
        self.drift_visits = drift_visits
        self.dxs = dxs
        self.ttype = ttype
        self.xcs = {}
        self.fiberId = {}
        self.imageWidth = None

        if dataIds is None:
            dataIds = []
            for a in "brn":
                for s in [1, 2, 3, 4]:
                    dataIds.append(self.key(a, s))
        self.dataIds = dataIds

        # analysis arrays, indexed by the index into drift_visits
        def makeDicts():
            data = {}
            for did in dataIds:
                data[did] = {}

            return data

        self.lefts = makeDicts()          # sets of left-
        self.rights = makeDicts()         # .   and right-hand boundaries for each visit in drift_visits
        self.driftedModels = makeDicts()  # the models for each visit in drift_visits
        self.flats = makeDicts()          # the flats for each visit in drift_visits
        self.corrections = makeDicts()    # the per-visit flux correction for each visit in drift_visits
        self.profiles = makeDicts()       # motion profiles for each fiber for each visit in drift_visits

    @staticmethod
    def key(dataId):
        return f"{dataId['arm']}{dataId['spectrograph']}"

    def __len__(self):
        return len(self.drift_visits)

    def __str__(self):
        return f"\"{self.name}\" {self.home_visit} {self.drift_visits}"

    def getVisits(self):
        return [self.home_visit] + self.drift_visits

    def getDataSets(self, getAll=False):
        return [k for k, v in self.flats.items() if v or getAll]

    def getArms(self, getAll=False):
        return set({_[0] for _ in self.getDataSets(getAll)})

    def getSpectrographs(self, getAll=False):
        return set({int(_[1]) for _ in self.getDataSets(getAll)})

    def __contains__(self, dataId):
        return self.key(dataId) in self.getDataSets()


def driver(butler, driftSet, arm, spectrograph, step=0.5, verbose=1, doPlot=False, doCalculateModel=True):
    """
    E.g.
    driftSet = "Yuki-2024-03-11:SM2:1"
    flat = driftFlats.driver(butler, driftSet, 'n', spectrograph=2)

    N.b. E.g.
    reduceExposure.py /work/drp --calib=/work/drp/CALIB --rerun=rhl/tmp \
       --id visit=107478^107452^107465 \
        --config targetType=[ENGINEERING] isr.doDark=True isr.doFlat=False \
                 doBoxcarExtraction=True boxcarWidth=11 doMeasureLines=False doAdjustDetectorMap=False \
                 useCalexp=True
    """

    dataId = dict(visit=driftSet.home_visit, arm=arm, spectrograph=spectrograph)
    key = driftSet.key(dataId)

    if verbose:
        print(f"read data for {'%(visit)d %(arm)s%(spectrograph)d' % dataId}")

    home = butler.get("calexp", dataId)
    homeDetMap = butler.get("detectorMap_used", dataId)
    pfsConfig = butler.get("pfsConfig", dataId).select(spectrograph=dataId["spectrograph"],
                                                       targetType=driftSet.ttype)
    spec = butler.get("pfsArm", dataId)

    if verbose:
        print("buildFiberProfiles")
    fiberProfiles, spectra = buildFiberProfiles(home, homeDetMap, pfsConfig, spec)
    if verbose:
        print("estimateScatteredLight")
    scatteredLight = estimateScatteredLight(home, homeDetMap, fiberProfiles, spectra)

    for vset in range(len(driftSet)):
        dataId["visit"] = driftSet.drift_visits[vset]
        exp = butler.get("calexp", dataId)
        detMap = butler.get("detectorMap_used", dataId)

        exp.image -= scatteredLight

        if verbose:
            print(f"\t{vset} {dataId['visit']}  findSlitMotion")
        findSlitMotion(driftSet, key, vset, exp, detMap, pfsConfig, fiberProfiles,
                       verbose=verbose, doPlot=doPlot, title=f"{dataId}")

        if verbose:
            print(f"\t{vset} {dataId['visit']}  estimateTraceIllumination")
        estimateTraceIllumination(driftSet, key, vset, exp, rowwidth=300)

        if not doCalculateModel:
            print(f"\t{vset} {dataId['visit']}  skipping calculateModel")
        else:
            print(f"\t{vset} {dataId['visit']}  calculateModel")
            calculateModel(driftSet, key, vset, home, homeDetMap, fiberProfiles, spectra,
                           pfsConfig.fiberId, step, verbose=verbose)
        #
        # Add the scattered light to the model
        #
        driftedModel = driftSet.driftedModels[key][vset]
        flux_drift = np.nansum(driftedModel.array)
        flux_home = np.nansum(home.image.array)

        driftedModel.array += scatteredLight.array*flux_drift/flux_home

        print(f"\t{vset} {dataId['visit']}  calculateOneDriftFlat")
        calculateOneDriftFlat(driftSet, key, vset, exp)

    flat, flats = mergeOneDriftFlats(driftSet, key, extra=15)

    return flat


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


def findSlitMotion(driftSet, key, vset, exp, detMap, pfsConfig, fiberProfiles, rowwidth=200,
                   verbose=0, doPlot=False, title=""):
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
    driftSet.lefts[key][vset], driftSet.rights[key][vset] = left, right

    for i, fid in enumerate(fiberProfiles.fiberId):
        left[fid] = np.NaN
        right[fid] = np.NaN

        fhole = pfsConfig.select(fiberId=fid).fiberHole[0]
        if fhole in [308, 316, 336, 341, 360]:   # too close to the centre of the detector pair in b/r/m
            continue

        xc = detMap.getXCenter(fid, 2000)
        ixc = int(xc + 0.5)

        ixc0 = min([0, ixc - int(0.25*width)])
        ixc1 = max([ixc + int(0.25*width)+1, driftSet.imageWidth])

        lev = np.mean(profile[ixc0:ixc1])

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
            left[fid]  = brentq(prof, al, bl) if al >= 0 else np.NaN  # noqa: E221
            right[fid] = brentq(prof, ar, br) if br < len(prof) - 1 else np.NaN
        except ValueError as e:
            if verbose > 1:
                print(f"Failed for fiberId {fid} (fiberHole {fhole}): {e}")
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

    driftSet.xcs[key] = xc
    driftSet.fiberId[key] = fids

    driftSet.profiles[key][vset] = profile


def showTraceEdges(pfsConfig, driftSet, key, vset, doPlot=True, doPrint=True, title=""):
    """
    N.b. code duplication with the "real" code; refactor?
    """
    fids = driftSet.fiberId[key]
    left = driftSet.lefts[key][vset]
    right = driftSet.rights[key][vset]
    profile = driftSet.profiles[key][vset]
    xc = driftSet.xcs[key]

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
            ixc0 = min([0, ixc - int(0.25*width)])
            ixc1 = max([ixc + int(0.25*width)+1, driftSet.imageWidth])

            lev = np.mean(profile[ixc0:ixc1])

            al = max([0, xc[i] + 1.05*dx[0]])
            br = min([xc[i] + 1.05*dx[1], driftSet.imageWidth - 1])

            c = 0.9*lev
            plt.plot([al, br], [c, c], color=f"C{i}")

            plt.axvline(left[fid], color=f"C{i}", alpha=1, zorder=-1)
            plt.axvline(right[fid], color=f"C{i}", alpha=1, zorder=-1)

    if doPlot:
        plt.legend(ncol=2)


def estimateTraceIllumination(driftSet, key, vset, exp, rowwidth=300):
    """Estimate the correction to the effective illumination of each column

    We do this based on averaging the bands produced by the different fibres
    """
    profile = np.median(exp.image.array[2000-rowwidth//2:2000+rowwidth//2+1], axis=0)

    fids = driftSet.fiberId[key]
    left = driftSet.lefts[key][vset]
    right = driftSet.rights[key][vset]

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

    # Trim the left and right ends to be >= correction_min
    correction_min = 0.99
    for i in range(len(correction)):
        if correction[i] >= correction_min:
            i0 = i
            break
    for i in range(len(correction) - 1, 0, -1):
        if correction[i] >= correction_min:
            i1 = i
            break

    correction = correction[i0: i1 + 1]

    driftSet.corrections[key][vset] = correction


def calculateModel(driftSet, key, vset, calexp, detMap, fiberProfiles, spectra, fiberId, step=0.5, verbose=0):
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
    fids = driftSet.fiberId[key]
    correction = driftSet.corrections[key][vset]
    left = driftSet.lefts[key][vset]
    right = driftSet.rights[key][vset]
    xc = driftSet.xcs[key]

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

    spatialOffsets0 = detMap.getSpatialOffsets().copy()
    spectralOffsets0 = detMap.getSpectralOffsets().copy()

    driftedModel = None
    for i, dx in enumerate(dxs):
        if verbose:
            print(f"\t\tdx: {dx:6.1f} {int(100*i/(len(dxs) - 1))}%", end='\r', flush=True)

        detMap.setSlitOffsets(spatialOffsets0, spectralOffsets0)  # reset the DetectorMap

        bad = []
        for fid in fiberId:
            if True:  # workaround inability to offset part of fibre off the chip
                xCenter = detMap.getXCenter(fid) + dx
                if np.min(xCenter) < 0 or np.max(xCenter) >= detMap.bbox.endX - 1:
                    bad.append(fid)
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
            driftSet.driftedModels[key][vset] = driftedModel
        else:
            driftedModel += image

    detMap.setSlitOffsets(spatialOffsets0, spectralOffsets0)
    print("")

    driftedModel /= len(dxs)
    # we don't want to divide by the background model, so set missing pixels to NaN before we add it
    driftedModel.array[driftedModel.array == 0] = np.NaN


def calculateOneDriftFlat(driftSet, key, vset, exp):
    """Calculate the flat from a single drift exposure
    """

    flat = exp.clone()
    driftSet.flats[key][vset] = flat

    driftedModel = driftSet.driftedModels[key][vset]

    flat.image /= driftedModel
    flat.image /= np.nanmedian(flat.image.array)


def mergeOneDriftFlats(driftSet, key, extra=15):
    nflat = len(driftSet.driftedModels[key])

    flats = driftSet.flats[key].copy()
    for i in range(nflat):
        flats[i] = driftSet.flats[key][i].clone()

    #  avoid extra pixels in the region near the left/right edges where the profile's rolling off

    margin = 3
    for vset in range(nflat):
        for left, right in zip(driftSet.lefts[key][vset].values(), driftSet.rights[key][vset].values()):
            il, ir = int(left + margin), int(right - margin)

            ile = max([il - extra, 0])
            ire = min([ir + extra, driftSet.imageWidth - 1])
            il = max([il, 0])
            ir = min([ir, driftSet.imageWidth - 1])

            flats[vset].image.array[:, ile:il] = np.NaN
            flats[vset].image.array[:, ir:ire] = np.NaN

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN slice encountered
        flat = np.nanmedian(np.array([_.image.array for _ in flats.values()]), axis=0).astype(np.float32)

    flat /= np.nanmedian(flat)

    return afwImage.ImageF(flat), flats
