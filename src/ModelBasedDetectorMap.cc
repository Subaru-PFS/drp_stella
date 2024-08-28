#include "ndarray.h"
#include "lsst/pex/exceptions.h"

#include "lsst/cpputils/Cache.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


ModelBasedDetectorMap::ModelBasedDetectorMap(
    lsst::geom::Box2I const& bbox,
    double wavelengthCenter,
    double dispersion,
    double sampling,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    Spline::ExtrapolationTypes extrapolation,
    double precision
) : DetectorMap(bbox, fiberId, spatialOffsets, spectralOffsets, visitInfo, metadata),
    _wavelengthCenter(wavelengthCenter),
    _dispersion(dispersion),
    _sampling(sampling),
    _precision(precision),
    _splines(fiberId.size()),
    _extrapolation(extrapolation)
    {}


void ModelBasedDetectorMap::_resetSlitOffsets() {
    _clearSplines();
    DetectorMap::_resetSlitOffsets();
}


ModelBasedDetectorMap::SplineGroup ModelBasedDetectorMap::_makeSplines(int fiberId) const {
    assert(_sampling > 0);  // to prevent infinite loops
    std::size_t const height = getBBox().getHeight();
    std::vector<SplineCoeffT> wavelength;
    std::vector<SplineCoeffT> xx;
    std::vector<SplineCoeffT> yy;

    wavelength.reserve(height);
    xx.reserve(height);
    yy.reserve(height);

    lsst::geom::Point2D point;  // Position on detector

    // Attempt to find a wavelength that's on the detector
    double startWavelength = std::numeric_limits<double>::quiet_NaN();
    for (float factor = 0.0; factor <= 1.0 && !std::isfinite(startWavelength); factor += 0.2) {
        for (bool negative : {false, true}) {
            double const wl = _wavelengthCenter + factor*0.5*height*_dispersion*(negative ? -1 : 1);
            try {
                point = evalModel(fiberId, wl);
            } catch (...) {
                continue;
            }
            if (!std::isfinite(point.getX()) || !std::isfinite(point.getY())) {
                continue;
            }
            startWavelength = wl;
            break;
        }
    }
    if (!std::isfinite(startWavelength)) {
        std::ostringstream msg;
        msg << "Unable to find good starting wavelength for fiberId=" << fiberId;
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, msg.str());
    }

    // Iterate up in wavelength until we drop off the edge of the detector
    // We want to get right up to the edge of the detector, if we can.
    // In order to do so, we'll pull back on the wavelength sampling if we go off the edge.
    SplineCoeffT goodWavelength = startWavelength;
    SplineCoeffT wlStep = _dispersion*_sampling;
    while (wlStep >= _dispersion*_precision) {
        double const wl = goodWavelength + wlStep;
        try {
            point = evalModel(fiberId, wl);
        } catch (...) {
            wlStep /= 2;
            continue;
        }
        if (!std::isfinite(point.getX()) || !std::isfinite(point.getY())) {
            wlStep /= 2;
            continue;
        }
        wavelength.push_back(wl);
        xx.push_back(point.getX());
        yy.push_back(point.getY());
        if (point.getY() > height - 0.5 || point.getY() < -0.5) {
            break;
        }
        goodWavelength = wl;
    }
    // Iterate down in wavelength until we drop off the edge of the detector
    wlStep = _dispersion*_sampling;
    goodWavelength = startWavelength;
    while (wlStep >= _dispersion*_precision) {
        double const wl = goodWavelength - wlStep;
        try {
            point = evalModel(fiberId, wl);
        } catch (...) {
            wlStep /= 2;
            continue;
        }
        if (!std::isfinite(point.getX()) || !std::isfinite(point.getY())) {
            wlStep /= 2;
            continue;
        }
        wavelength.push_back(wl);
        xx.push_back(point.getX());
        yy.push_back(point.getY());
        if (point.getY() < -0.5 || point.getY() > height - 0.5) {
            break;
        }
        goodWavelength = wl;
    }
    std::size_t const length = wavelength.size();
    if (length < 3) {
        std::ostringstream msg;
        msg << "Insufficient good points for fiberId=" << fiberId;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, msg.str());
    }

    // Sort into monotonic ndarrays
    // With some care we could simply rearrange, but easier to code the sort
    // and performance isn't critical.
    ndarray::Array<std::size_t, 1, 1> indices = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        indices[ii] = ii;
    }
    std::sort(indices.begin(), indices.end(),
                [&yy](std::size_t left, std::size_t right) { return yy[left] < yy[right]; });

    ndarray::Array<SplineCoeffT, 1, 1> wlArray = ndarray::allocate(length);
    ndarray::Array<SplineCoeffT, 1, 1> xArray = ndarray::allocate(length);
    ndarray::Array<SplineCoeffT, 1, 1> yArray = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        std::size_t const index = indices[ii];
        wlArray[ii] = wavelength[index];
        xArray[ii] = xx[index];
        yArray[ii] = yy[index];
    }

    auto const interpolation = Spline::CUBIC_NOTAKNOT;
    return std::make_tuple(
        Spline(yArray, wlArray, interpolation, _extrapolation),
        Spline(yArray, xArray, interpolation, _extrapolation),
        Spline(wlArray, yArray, interpolation, _extrapolation)
    );
}


lsst::geom::Point2D ModelBasedDetectorMap::findPointImpl(int fiberId, double wavelength) const {
    try {
        double const yy = _getRowSpline(fiberId)(wavelength);
        if (!std::isfinite(yy)) {
            return lsst::geom::Point2D(std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN());
        }
        double const xx = _getXCenterSpline(fiberId)(yy);
        return lsst::geom::Point2D(xx, yy);
    } catch (...) {
        return lsst::geom::Point2D(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::quiet_NaN());
    }
}


double ModelBasedDetectorMap::getXCenterImpl(int fiberId, double row) const {
    try {
        return _getXCenterSpline(fiberId)(row);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}


double ModelBasedDetectorMap::findWavelengthImpl(int fiberId, double row) const {
    try {
        return _getWavelengthSpline(fiberId)(row);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}


}}}  // namespace pfs::drp::stella
