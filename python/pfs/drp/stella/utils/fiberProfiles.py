import lsst.afw.image as afwImage
from pfs.drp.stella.FiberTrace import fiberMaskPlane

__all__ = ["makeFiberTraceMaskedImage"]


def makeFiberTraceMaskedImage(bbox, fiberTraceSet, stride=1):
    """Return a MaskedImage with the given BBox for a fiberTraceSet

    bbox: `lsst.geom.Box2I` Image bounding box
    fiberTraceSet: `pfs.drp.stella.FiberTraceSet` The fiber profiles
    stride: `int` Only plot a subset of fibres, e.g. every third for stride=3

    E.g.
       calexp = butler.get("calexp", dataId)
       fiberProfiles = butler.get("fiberProfiles", dataId)
       fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)

       profileIm = makeFiberTraceMaskedImage(calexp.getBBox(), fiberTraces)
    """
    mi = afwImage.MaskedImageF(bbox)
    mi.mask.addMaskPlane(fiberMaskPlane)

    for i, ft in enumerate(fiberTraceSet):
        if i%stride == 0:
            ftMask = ft.trace
            mi[ftMask.getBBox()] += ftMask

    return mi
