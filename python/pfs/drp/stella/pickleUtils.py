""" Allow pickling LSST objects that don't natively support it

This covers `lsst.afw.image.VisitInfo` (and its components), and persistable
`lsst.afw.math` kernels and functions.
"""
import copyreg
from lsst.afw.image import VisitInfo
from lsst.afw.coord import Observatory, Weather
from lsst.afw.fits import MemFileManager
import lsst.afw.math as afwMath


def pickleVisitInfo(info):
    return (VisitInfo,
            tuple(getattr(info, "get" + prop)() for
                  prop in ("Id", "ExposureTime", "DarkTime", "Date", "Ut1", "Era", "BoresightRaDec",
                           "BoresightAzAlt", "BoresightAirmass", "BoresightRotAngle", "RotType",
                           "Observatory", "Weather")))


def pickleObservatory(obs):
    return (Observatory, (obs.getLongitude(), obs.getLatitude(), obs.getElevation()))


def pickleWeather(weather):
    return (Weather, (weather.getAirTemperature(), weather.getAirPressure(), weather.getHumidity()))


copyreg.pickle(VisitInfo, pickleVisitInfo)
copyreg.pickle(Observatory, pickleObservatory)
copyreg.pickle(Weather, pickleWeather)


def unpicklePersistable(cls, data, size):
    """Unpickle a persistable object from its in-memory FITS representation

    Parameters
    ----------
    cls : `type`
        Class of the object; must have a ``readFits`` class method accepting
        a `lsst.afw.fits.MemFileManager`.
    data : `bytes`
        In-memory FITS representation of the object.
    size : `int`
        Length of ``data``.

    Returns
    -------
    obj : ``cls``
        Unpickled object.
    """
    manager = MemFileManager(size)
    manager.setData(data, size)
    return cls.readFits(manager)


def picklePersistable(obj):
    """Pickle a persistable object via its in-memory FITS representation

    This differs from `lsst.afw.fits.reduceToFits` in that it doesn't
    attempt to provide compression options to ``writeFits``, which
    `lsst.afw.table.io.Persistable` doesn't accept.

    Parameters
    ----------
    obj : `lsst.afw.table.io.Persistable`
        Object to pickle.

    Returns
    -------
    reduced : `tuple` [callable, `tuple`]
        Tuple in the format returned by ``__reduce__``.
    """
    if not obj.isPersistable():
        raise TypeError(f"Cannot pickle non-persistable {type(obj).__name__}")
    manager = MemFileManager()
    obj.writeFits(manager)
    return (unpicklePersistable, (type(obj), manager.getData(), manager.getLength()))


for cls in (
    afwMath.AnalyticKernel,
    afwMath.DeltaFunctionKernel,
    afwMath.FixedKernel,
    afwMath.LinearCombinationKernel,
    afwMath.SeparableKernel,
    afwMath.Chebyshev1Function2D,
    afwMath.GaussianFunction2D,
    afwMath.DoubleGaussianFunction2D,
    afwMath.PolynomialFunction2D,
):
    copyreg.pickle(cls, picklePersistable)
