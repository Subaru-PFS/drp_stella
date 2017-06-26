import lsst.afw.image as afwImage
import pfs.drp.stella as drpStella

flatfile = "/Users/azuri/spectra/pfs/2015-04-24/IR-flatFieldx10-0-23-noNoise-constantPsf.fz"
combfile = "/Users/azuri/spectra/pfs/2015-04-24/IR-combFieldx10-0-23-noNoise-constantPsf.fz"

flat = afwImage.ImageF(flatfile)
flat = afwImage.makeExposure(afwImage.makeMaskedImage(flat))
biasFlat = afwImage.ImageF(flatfile, 4)
flat.getMaskedImage()[:] -= biasFlat
flat.getMaskedImage().getVariance().getArray()[:][:] = flat.getMaskedImage().getImage().getArray()[:][:]
comb = afwImage.ImageF(combfile)
comb = afwImage.makeExposure(afwImage.makeMaskedImage(comb))
bias = afwImage.ImageF(combfile, 4)
comb.getMaskedImage()[:] -= bias
comb.getMaskedImage().getVariance().getArray()[:][:] = comb.getMaskedImage().getImage().getArray()[:][:]

ftffc = drpStella.FiberTraceFunctionFindingControl()
ftffc.signalThreshold = 10.
ftffc.minLength = 1000
ftffc.fiberTraceFunctionControl.interpolation = "CHEBYSHEV"
ftffc.fiberTraceFunctionControl.order = 5
weightBase = 10000.
iTrace = 0
regularization = 0.00005
xCorRangeLowLimit = -0.1
xCorRangeHighLimit = 0.1
xCorStepSize = 0.001
weightedTPS = False
swathWidth = 500
meanOnly = False
iFT = 0

fts = drpStella.findAndTraceAperturesF(flat.getMaskedImage(), ftffc)
