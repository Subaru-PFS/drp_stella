#include "ndarray.h"

#include "pfs/drp/stella/ModelBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


ModelBasedDetectorMap::ModelBasedDetectorMap(
    lsst::geom::Box2I const& bbox,
    double wavelengthCenter,
    double wavelengthSampling,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : DetectorMap(bbox, fiberId, spatialOffsets, spectralOffsets, visitInfo, metadata),
    _wavelengthCenter(wavelengthCenter),
    _wavelengthSampling(wavelengthSampling),
    _splinesInitialized(false)
    {}


void ModelBasedDetectorMap::_setSplines() const {
    _rowToWavelength.clear();
    _rowToXCenter.clear();
    _rowToWavelength.reserve(getNumFibers());
    _rowToXCenter.reserve(getNumFibers());

    assert(_wavelengthSampling > 0);  // to prevent infinite loops
    std::size_t const height = getBBox().getHeight();
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        std::vector<SplineCoeffT> wavelength;
        std::vector<SplineCoeffT> xx;
        std::vector<SplineCoeffT> yy;

        wavelength.reserve(height);
        xx.reserve(height);
        yy.reserve(height);
        int fiberId = getFiberId()[ii];

        // Iterate up in wavelength until we drop off the edge of the detector
        for (SplineCoeffT wl = _wavelengthCenter; true; wl += _wavelengthSampling) {
            auto const point = findPointImpl(fiberId, wl);
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() > height || point.getY() < 0) {
                break;
            }
        }
        // Iterate down in wavelength until we drop off the edge of the detector
        for (SplineCoeffT wl = _wavelengthCenter - _wavelengthSampling; true; wl -= _wavelengthSampling) {
            auto const point = findPointImpl(fiberId, wl);
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() < 0 || point.getY() > height) {
                break;
            }
        }
        std::size_t const length = wavelength.size();

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
        _rowToWavelength.emplace_back(yArray, wlArray);
        _rowToXCenter.emplace_back(yArray, xArray);
    }
}


DetectorMap::Array1D ModelBasedDetectorMap::getXCenter(
    int fiberId
) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToXCenter[getFiberIndex(fiberId)];
    std::size_t const height = getBBox().getHeight();
    Array1D out = ndarray::allocate(height);
    for (std::size_t yy = 0; yy < height; ++yy) {
        out[yy] = spline(yy);
    }
    return out;
}


double ModelBasedDetectorMap::getXCenter(
    int fiberId,
    double row
) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToXCenter[getFiberIndex(fiberId)];
    return spline(row);
}


DetectorMap::Array1D ModelBasedDetectorMap::getWavelength(
    int fiberId
) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    std::size_t const height = getBBox().getHeight();
    Array1D out = ndarray::allocate(height);
    for (std::size_t yy = 0; yy < height; ++yy) {
        out[yy] = spline(yy);
    }
    return out;
}


double ModelBasedDetectorMap::getWavelength(
    int fiberId,
    double row
) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    return spline(row);
}


int ModelBasedDetectorMap::findFiberId(lsst::geom::PointD const& point) const {
    if (getNumFibers() == 1) {
        return getFiberId()[0];
    }
    SplineCoeffT const xx = point.getX();
    SplineCoeffT const yy = point.getY();

    _ensureSplinesInitialized();

    // We know x as a function of fiberId (given y),
    // and x is monotonic with fiberId (for fixed y),
    // so we can find fiberId by bisection.
    std::size_t lowIndex = 0;
    std::size_t highIndex = getNumFibers() - 1;
    SplineCoeffT xLow = _rowToXCenter[lowIndex](yy);
    SplineCoeffT xHigh = _rowToXCenter[highIndex](yy);
    bool const increasing = xHigh > xLow;  // Does x increase with increasing fiber index?
    while (highIndex - lowIndex > 1) {
        std::size_t newIndex = lowIndex + (highIndex - lowIndex)/2;
        SplineCoeffT xNew = _rowToXCenter[newIndex](yy);
        if (increasing) {
            assert(xNew > xLow && xNew < xHigh);
            if (xx > xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        } else {
            assert(xNew < xLow && xNew > xHigh);
            if (xx < xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        }
    }
    return std::abs(xx - xLow) < std::abs(xx - xHigh) ? getFiberId()[lowIndex] : getFiberId()[highIndex];
}


double ModelBasedDetectorMap::findWavelengthImpl(int fiberId, double row) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    return spline(row);
}


void ModelBasedDetectorMap::_resetSlitOffsets() {
    _splinesInitialized = false;
}


void ModelBasedDetectorMap::_ensureSplinesInitialized() const {
    if (_splinesInitialized) return;
    _setSplines();
    _splinesInitialized = true;
}


}}}  // namespace pfs::drp::stella
