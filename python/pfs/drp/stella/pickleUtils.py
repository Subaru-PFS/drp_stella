""" Allow pickling an lsst.afw.image.VisitInfo"""
import copyreg
from lsst.afw.image import VisitInfo
from lsst.afw.coord import Observatory, Weather


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
