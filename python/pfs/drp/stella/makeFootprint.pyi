from lsst.afw.detection import Footprint
from lsst.afw.image import ImageF
from lsst.geom import Point2I


def makeFootprint(image: ImageF, peak: Point2I, height: int, width: float) -> Footprint: ...
