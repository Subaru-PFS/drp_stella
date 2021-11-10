#include "ndarray.h"
#include "lsst/pex/exceptions.h"

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
        int const fiberId = getFiberId()[ii];

        lsst::geom::Point2D point;  // Position on detector

        // Iterate up in wavelength until we drop off the edge of the detector
        for (SplineCoeffT wl = _wavelengthCenter; true; wl += _wavelengthSampling) {
            try {
                point = findPointImpl(fiberId, wl);
            } catch (...) {
                break;
            }
            if (!std::isfinite(point.getX()) || !std::isfinite(point.getY())) {
                break;
            }
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() > height || point.getY() < 0) {
                break;
            }
        }
        // Iterate down in wavelength until we drop off the edge of the detector
        for (SplineCoeffT wl = _wavelengthCenter - _wavelengthSampling; true; wl -= _wavelengthSampling) {
            try {
                point = findPointImpl(fiberId, wl);
            } catch (...) {
                break;
            }
            if (!std::isfinite(point.getX()) || !std::isfinite(point.getY())) {
                break;
            }
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() < 0 || point.getY() > height) {
                break;
            }
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
        _rowToWavelength.emplace_back(yArray, wlArray);
        _rowToXCenter.emplace_back(yArray, xArray);
    }
}


double ModelBasedDetectorMap::getXCenterImpl(
    int fiberId,
    double row
) const {
    _ensureSplinesInitialized();
    Spline const& spline = _rowToXCenter[getFiberIndex(fiberId)];
    return spline(row);
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
