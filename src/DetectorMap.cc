#include <algorithm>

#include "pfs/drp/stella/utils/checkSize.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {


DetectorMap::DetectorMap(
    lsst::geom::Box2I bbox,
    DetectorMap::FiberIds const& fiberId,
    DetectorMap::Array1D const& spatialOffsets,
    DetectorMap::Array1D const& spectralOffsets,
    DetectorMap::VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _bbox(bbox),
    _fiberId(ndarray::allocate(fiberId.size())),
    _spatialOffsets(ndarray::allocate(fiberId.size())),
    _spectralOffsets(ndarray::allocate(fiberId.size())),
    _visitInfo(visitInfo),
    _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
{
    // Ensure the fiberIds are sorted, since some algorithms depend on that.
    ndarray::Array<std::size_t, 1, 1> indices = ndarray::allocate(getNumFibers());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        indices[ii] = ii;
    }
    std::sort(indices.begin(), indices.end(),
              [&fiberId](std::size_t left, std::size_t right) { return fiberId[left] < fiberId[right]; });
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        _fiberId[ii] = fiberId[indices[ii]];
        _fiberMap[_fiberId[ii]] = ii;
    }
    if (spatialOffsets.isEmpty()) {
        _spatialOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spatialOffsets.size(), fiberId.size(), "spatialOffsets");
        for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
            _spatialOffsets[ii] = spatialOffsets[indices[ii]];
        }
    }
    if (spectralOffsets.isEmpty()) {
        _spectralOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spectralOffsets.size(), fiberId.size(), "spectralOffsets");
        for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
            _spectralOffsets[ii] = spectralOffsets[indices[ii]];
        }
    }
}


void DetectorMap::applySlitOffset(
    double spatial,
    double spectral
) {
    if (spatial != 0.0) {
        std::for_each(_spatialOffsets.begin(), _spatialOffsets.end(),
                      [spatial](double& value) { value += spatial; });
    }
    if (spectral != 0.0) {
        std::for_each(_spectralOffsets.begin(), _spectralOffsets.end(),
                      [spectral](double& value) { value += spectral; });
    }
    _resetSlitOffsets();
}


void DetectorMap::setSlitOffsets(
    DetectorMap::Array1D const& spatial,
    DetectorMap::Array1D const& spectral
) {
    utils::checkSize(spatial.size(), getNumFibers(), "spatial");
    utils::checkSize(spectral.size(), getNumFibers(), "spectral");
    _spatialOffsets.deep() = spatial;
    _spectralOffsets.deep() = spectral;
    _resetSlitOffsets();
}


void DetectorMap::setSlitOffsets(
    int fiberId,
    double spatial,
    double spectral
) {
    std::size_t const index = getFiberIndex(fiberId);
    _spatialOffsets[index] = spatial;
    _spectralOffsets[index] = spectral;
    _resetSlitOffsets();
}


DetectorMap::Array2D DetectorMap::getWavelength() const {
    Array2D wavelength = ndarray::allocate(getNumFibers(), _bbox.getHeight());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        wavelength[ii].deep() = getWavelength(_fiberId[ii]);
    }
    return wavelength;
}


DetectorMap::Array1D DetectorMap::getWavelength(
    int fiberId
) const {
    std::size_t const height = getBBox().getHeight();
    ndarray::Array<double, 1, 1> out = ndarray::allocate(height);
    for (std::size_t yy = 0; yy < height; ++yy) {
        out[yy] = findWavelengthImpl(fiberId, yy);
    }
    return out;
}


double DetectorMap::getDispersionAtCenter(int fiberId) const {
    float const delta = 0.5;
    int const center = 0.5*_bbox.getHeight();
    double const wl0 = findWavelength(fiberId, center - delta);
    double const wl1 = findWavelength(fiberId, center + delta);
    return std::abs(wl1 - wl0)/(2*delta);
}


double DetectorMap::getDispersion(int fiberId, double wavelength, double dWavelength) const {
    double const y0 = findPoint(fiberId, wavelength - 0.5*dWavelength).getY();
    double const y1 = findPoint(fiberId, wavelength + 0.5*dWavelength).getY();
    return dWavelength/(y1 - y0);
}


DetectorMap::Array1D DetectorMap::getDispersion(
    FiberIds const& fiberId,
    Array1D const& wavelength,
    double dWavelength
) const {
    std::size_t const num = fiberId.size();
    utils::checkSize(wavelength.size(), num, "fiberId vs wavelength");
    Array1D dispersion = ndarray::allocate(num);
    for (std::size_t ii = 0; ii < num; ++ii) {
        dispersion[ii] = getDispersion(fiberId[ii], wavelength[ii], dWavelength);
    }
    return dispersion;
}


DetectorMap::Array2D DetectorMap::getXCenter() const {
    Array2D xCenter = ndarray::allocate(getNumFibers(), _bbox.getHeight());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        xCenter[ii].deep() = getXCenter(_fiberId[ii]);
    }
    return xCenter;
}

DetectorMap::Array1D DetectorMap::getXCenter(int fiberId) const {
    int const height = getBBox().getHeight();
    Array1D xCenter = ndarray::allocate(height);
    for (int yy = 0; yy < height; ++yy) {
        xCenter[yy] = getXCenterImpl(fiberId, yy);
    }
    return xCenter;
}


DetectorMap::Array1D DetectorMap::getXCenter(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& row
) const {
    utils::checkSize(fiberId.size(), row.size(), "fiberId vs row");
    Array1D xCenter = ndarray::allocate(fiberId.size());
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        xCenter[ii] = getXCenter(fiberId[ii], row[ii]);
    }
    return xCenter;
}


DetectorMap::Array1D DetectorMap::getXCenter(
    int fiberId,
    ndarray::Array<double, 1, 1> const& row
) const {
    Array1D xCenter = ndarray::allocate(row.size());
    for (std::size_t ii = 0; ii < row.size(); ++ii) {
        xCenter[ii] = getXCenter(fiberId, row[ii]);
    }
    return xCenter;
}


lsst::geom::Point2D DetectorMap::findPoint(int fiberId, double wavelength, bool throwOnError) const {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
    lsst::geom::Point2D point;
    try {
        point = findPointImpl(fiberId, wavelength);
    } catch (...) {
        if (throwOnError) {
            throw;
        }
        return lsst::geom::PointD(NaN, NaN);
    }
    if (point.getX() < _bbox.getMinX() || point.getX() > _bbox.getMaxX() ||
        point.getY() < _bbox.getMinY() || point.getY() > _bbox.getMaxY()) {
            return lsst::geom::PointD(NaN, NaN);
    }
    return point;
}


ndarray::Array<double, 2, 1> DetectorMap::findPoint(
    int fiberId,
    ndarray::Array<double, 1, 1> const& wavelength
) const {
    std::size_t const length = wavelength.size();
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = findPoint(fiberId, wavelength[ii]);
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
}


ndarray::Array<double, 2, 1> DetectorMap::findPoint(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength
) const {
    std::size_t const length = fiberId.size();
    utils::checkSize(length, wavelength.size(), "fiberId vs wavelength");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = findPoint(fiberId[ii], wavelength[ii]);
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
}


double DetectorMap::findWavelength(int fiberId, double row, bool throwOnError) const {
    try {
        return findWavelengthImpl(fiberId, row);
    } catch (...) {
        if (throwOnError) {
            throw;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
}


ndarray::Array<double, 1, 1> DetectorMap::findWavelength(
    int fiberId,
    ndarray::Array<double, 1, 1> const& row
) const {
    std::size_t const length = row.size();
    ndarray::Array<double, 1, 1> out = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        out[ii] = findWavelength(fiberId, row[ii]);
    }
    return out;
}


ndarray::Array<double, 1, 1> DetectorMap::findWavelength(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& row
) const {
    std::size_t const length = fiberId.size();
    utils::checkSize(length, row.size(), "fiberId vs row");
    ndarray::Array<double, 1, 1> out = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        out[ii] = findWavelength(fiberId[ii], row[ii]);
    }
    return out;
}


int DetectorMap::findFiberId(
    lsst::geom::PointD const& point
) const {
    double const xx = point.getX();
    double const yy = point.getY();

    if (!getBBox().contains(lsst::geom::PointI(xx, yy))) {
        std::ostringstream os;
        os << "Point " << point << " does not lie within BBox " << getBBox();
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainError, os.str());
    }
    auto const fiberId = getFiberId();
    if (getNumFibers() == 1) {
        return fiberId[0];
    }

    // We know x as a function of fiberId (given y),
    // and x is monotonic with fiberId (for fixed y),
    // so we can find fiberId by bisection.
    std::size_t lowIndex = 0;
    std::size_t highIndex = getNumFibers() - 1;
    double xLow = getXCenterImpl(fiberId[lowIndex], yy);
    double xHigh = getXCenterImpl(fiberId[highIndex], yy);
    bool const increasing = xHigh > xLow;  // Does x increase with increasing fiber index?
    while (highIndex - lowIndex > 1) {
        std::size_t const newIndex = lowIndex + (highIndex - lowIndex)/2;
        double const xNew = getXCenterImpl(fiberId[newIndex], yy);
        if (increasing) {
            if (xNew < xLow || xNew > xHigh) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Bisection failed");
            }
            if (xx > xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        } else {
            if (xNew > xLow || xNew < xHigh) {
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Bisection failed");
            }
            if (xx < xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        }
    }
    return std::abs(xx - xLow) < std::abs(xx - xHigh) ? fiberId[lowIndex] : fiberId[highIndex];
}


}}}  // namespace pfs::drp::stella
