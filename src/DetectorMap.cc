#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/daf/base.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {

DetectorMap::DetectorMap(
    lsst::geom::Box2I bbox,
    FiberMap const& fiberIds,
    std::vector<ndarray::Array<float, 1, 1>> const& centerKnots,
    std::vector<ndarray::Array<float, 1, 1>> const& centerValues,
    std::vector<ndarray::Array<float, 1, 1>> const& wavelengthKnots,
    std::vector<ndarray::Array<float, 1, 1>> const& wavelengthValues,
    Array2D const& slitOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _nFiber(fiberIds.getShape()[0]),
    _bbox(bbox),
    _fiberIds(ndarray::copy(fiberIds)),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(ndarray::makeVector(std::size_t(3), _nFiber)),
    _visitInfo(visitInfo),
    _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
{
    utils::checkSize(centerKnots.size(), _nFiber, "DetectorMap: centerKnots");
    utils::checkSize(centerValues.size(), _nFiber, "DetectorMap: centerValues");
    utils::checkSize(wavelengthKnots.size(), _nFiber, "DetectorMap: wavelengthKnots");
    utils::checkSize(wavelengthValues.size(), _nFiber, "DetectorMap: wavelengthValues");
    if (!slitOffsets.isEmpty()) {
        setSlitOffsets(slitOffsets);   // actually this is where we check slitOffsets
    } else {
        _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
    }

    for (std::size_t ii = 0; ii < _nFiber; ++ii) {
        float const minCenterKnot = *std::min_element(centerKnots[ii].begin(), centerKnots[ii].end());
        float const maxCenterKnot = *std::max_element(centerKnots[ii].begin(), centerKnots[ii].end());
        float const minWavelengthKnot = *std::min_element(wavelengthKnots[ii].begin(),
                                                          wavelengthKnots[ii].end());
        float const maxWavelengthKnot = *std::max_element(wavelengthKnots[ii].begin(),
                                                          wavelengthKnots[ii].end());
        if (minCenterKnot < bbox.getMinY() || maxCenterKnot > bbox.getMaxY()) {
            std::ostringstream os;
            os << "centerKnots[" << ii << "] out of range of bbox: " <<
                minCenterKnot << ".." << maxCenterKnot << " vs " << bbox;
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
        }
        if (minWavelengthKnot < bbox.getMinY() || maxWavelengthKnot > bbox.getMaxY()) {
            std::ostringstream os;
            os << "wavelengthKnots[" << ii << "] out of range of bbox: " <<
                minWavelengthKnot << ".." << maxWavelengthKnot << " vs " << bbox;
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
        }
        _yToXCenter[ii] = std::make_shared<math::Spline<float>>(centerKnots[ii], centerValues[ii]);
        _yToWavelength[ii] = std::make_shared<math::Spline<float>>(wavelengthKnots[ii], wavelengthValues[ii]);
    }
    _set_xToFiberId();
}

            
/*
 * Return a fiberIdx given a fiberId
 */
std::size_t
DetectorMap::getFiberIndex(int fiberId) const
{
    auto el = std::lower_bound(_fiberIds.begin(), _fiberIds.end(), fiberId);
    if (el == _fiberIds.end() || *el != fiberId) {
        std::ostringstream os;
        os << "Unknown fiberId " << fiberId;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    return el - _fiberIds.begin();      // index into _fiberId
}

/*
 * Set the offsets of the wavelengths of each fibre (in floating-point pixels)
 */
void
DetectorMap::setSlitOffsets(ndarray::Array<float, 2, 1> const& slitOffsets)
{
    if (slitOffsets.getShape()[0] != 3) {
        std::ostringstream os;
        os << "Number of types of offsets == " << slitOffsets.getShape()[0] <<
            ".  Expected three (dx, dy, dfocus)";
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
    if (slitOffsets.getShape()[1] != _nFiber) {
        std::ostringstream os;
        os << "Number of offsets == " << slitOffsets.getShape()[1] <<
            " != number of fibres == " << _nFiber;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }

    _slitOffsets.deep() = slitOffsets;
}

math::Spline<float> const&
DetectorMap::getWavelengthSpline(std::size_t index) const {
    if (index < 0 || index >= _nFiber) {
        std::ostringstream os;
        os << "Fiber index " << index << " out of range [" << 0 << "," << _nFiber << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    auto spline = _yToWavelength[index];
    if (!spline) {
        std::ostringstream os;
        os << "Wavelength spline for fiber index " << index << " not set";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    return *spline;
}

math::Spline<float> const&
DetectorMap::getCenterSpline(std::size_t index) const {
    if (index < 0 || index >= _nFiber) {
        std::ostringstream os;
        os << "Fiber index " << index << " out of range [" << 0 << "," << _nFiber << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    auto spline = _yToXCenter[index];
    if (!spline) {
        std::ostringstream os;
        os << "Center spline for fiber index " << index << " not set";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    return *spline;
}

/*
 * Return the wavelength values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getWavelength(std::size_t fiberId) const
{
    std::size_t const index = getFiberIndex(fiberId);
    auto const & spline = getWavelengthSpline(index);
    const float slitOffsetY = _slitOffsets[DY][index];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY);
    }

    return res;
}

std::vector<DetectorMap::Array1D> DetectorMap::getWavelength() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    std::vector<Array1D> result{numFibers};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii] = ndarray::copy(getWavelength(_fiberIds[ii]));
    }
    return result;
}


void DetectorMap::setWavelength(
    std::size_t fiberId,
    Array1D const& wavelength
) {
    Array1D rows = ndarray::allocate(_bbox.getHeight());
    int ii = _bbox.getMinY();
    for (auto rr = rows.begin(); rr != rows.end(); ++rr, ++ii) {
        *rr = ii;
    }
    setWavelength(fiberId, rows, wavelength);
}


/*
 * Return the wavelength values for a fibre
 */
void
DetectorMap::setWavelength(std::size_t fiberId,
                           ndarray::Array<float, 1, 1> const& knots,
                           ndarray::Array<float, 1, 1> const& wavelength)
{
    int const index = getFiberIndex(fiberId);
    _yToWavelength[index] = std::make_shared<Spline>(knots, wavelength);
}

/*
 * Return the xCenter values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getXCenter(std::size_t fiberId) const
{
    int const index = getFiberIndex(fiberId);
    auto const & spline = getCenterSpline(index);
    const float slitOffsetX = _slitOffsets[DX][index];
    const float slitOffsetY = _slitOffsets[DY][index];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY) + slitOffsetX;
    }

    return res;
}

std::vector<DetectorMap::Array1D> DetectorMap::getXCenter() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    std::vector<Array1D> result{numFibers};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii] = ndarray::copy(getXCenter(_fiberIds[ii]));
    }
    return result;
}

/*
 * Return the xCenter values for a fibre
 */
float
DetectorMap::getXCenter(std::size_t fiberId, float y) const
{
    int const index = getFiberIndex(fiberId);
    auto const & spline = getCenterSpline(index);
    const float slitOffsetX = _slitOffsets[DX][index];
    const float slitOffsetY = _slitOffsets[DY][index];

    return spline(y - slitOffsetY) + slitOffsetX;
}


void DetectorMap::setXCenter(
    std::size_t fiberId,
    Array1D const& xCenter
) {
    Array1D rows = ndarray::allocate(_bbox.getHeight());
    int ii = _bbox.getMinY();
    for (auto rr = rows.begin(); rr != rows.end(); ++rr, ++ii) {
        *rr = ii;
    }
    setXCenter(fiberId, rows, xCenter);
}


/*
 * Update the xcenter values for a fibre
 */
void
DetectorMap::setXCenter(std::size_t fiberId,
                        ndarray::Array<float, 1, 1> const& knots,
                        ndarray::Array<float, 1, 1> const& xCenter)
{
    int const index = getFiberIndex(fiberId);
    _yToXCenter[index] = std::make_shared<Spline>(knots, xCenter);
}

/*
 * Return the position of the fiber trace on the detector, given a fiberId and wavelength
 */
lsst::geom::PointD
DetectorMap::findPoint(int fiberId,               ///< Desired fibreId
                       float wavelength           ///< desired wavelength
                      ) const
{
    auto fiberWavelength = getWavelength(fiberId);

    auto begin = fiberWavelength.begin();
    auto end = fiberWavelength.end();
    auto onePast = std::lower_bound(begin, end, wavelength); // pointer to element just larger than wavelength

    if (onePast == end) {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::PointD(NaN, NaN);
    }

    auto fiberXCenter = getXCenter(fiberId);
    const auto iy = onePast - begin;

    double x = fiberXCenter[iy];
    double y = iy;                      // just past the desired point

    if (iy > 0) {                       // interpolate to get better accuracy
        const float dy = -(fiberWavelength[iy] - wavelength)/(fiberWavelength[iy] - fiberWavelength[iy - 1]);
        x += dy*(fiberXCenter[iy] - fiberXCenter[iy - 1]);
        y += dy;
    }

    return lsst::geom::PointD(x, y);
}

/************************************************************************************************************/
/*
 * Return the position of the fiber trace on the detector, given a fiberId and wavelength
 */
float
DetectorMap::findWavelength(int fiberId,               ///< Desired fibreId
                            float row                  ///< desired row
                           ) const
{
    if (row < 0 || row > _bbox.getHeight() - 1) {
        std::ostringstream os;
        os << "Row " << row << " is not in range [0, " << _bbox.getHeight() - 1 << "]";
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    int const index = getFiberIndex(fiberId);
    auto const & spline = getWavelengthSpline(index);

    return spline(row - getSlitOffsets(fiberId)[DY]);
}

/************************************************************************************************************/
/*
 * Return the fiberId given a position on the detector
 */
int
DetectorMap::findFiberId(lsst::geom::PointD pixelPos // position on detector
                          ) const
{
    float const x = pixelPos[0], y = pixelPos[1];
    std::size_t const maxIter = 2*_fiberIds.getNumElements();  // maximum number of iterations

    if (!_bbox.contains(lsst::geom::PointI(x, y))) {
        std::ostringstream os;
        os << "Point " << pixelPos << " does not lie within BBox " << _bbox;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }

    int fiberId = _xToFiberId[int(x)];    // first guess at the fiberId
    std::size_t index = getFiberIndex(fiberId);
    std::size_t iterations = 0;
    for (bool updatedIndex = true; updatedIndex; ++iterations) {
        auto const & spline0 = getCenterSpline(index);
        float const xCenter0 = spline0(y - getSlitOffsets()[DY][index]) + getSlitOffsets()[DX][index];

        updatedIndex = false;
        if (index > 0) {
            auto const & splineMinus = getCenterSpline(index - 1);
            float const xCenterMinus = splineMinus(y - getSlitOffsets()[DY][index - 1]) +
                getSlitOffsets()[DX][index - 1];

            if (std::fabs(x - xCenterMinus) < std::fabs(x - xCenter0)) {
                --index;
                updatedIndex = true;
                continue;
            }
        }
        if (index < _yToXCenter.size() - 1) {
            auto const & splinePlus = getCenterSpline(index + 1);
            float const xCenterPlus = splinePlus(y - getSlitOffsets()[DY][index]) +
                getSlitOffsets()[DX][index + 1];

            if (std::fabs(x - xCenterPlus) < std::fabs(x - xCenter0)) {
                ++index;
                updatedIndex = true;
                continue;
            }
        }
    }
    if (iterations > maxIter) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Exceeded maximum number of iterations");
    }

    return _fiberIds[index];
}

/************************************************************************************************************/

/*
 * set _xToFiberId, an array giving the fiber ID for each pixel across the centre of the chip
 */
void
DetectorMap::_set_xToFiberId()
{
    int const iy = _bbox.getHeight()/2;

    float lastCenter = -1e30;            // center of the last fibre we passed scanning to the right
    std::size_t lastIndex = -1;          // index of the last fibre we passed.  Special case starting with 0

    float nextCenter = getXCenter(_fiberIds[0], iy); // center of the next fiber we're looking at
    for (std::size_t index = 0, x = 0; x != std::size_t(_bbox.getWidth()); ++x) {
        if (x - lastCenter > nextCenter - x) { // x is nearer to next_xc than last_xc
            lastCenter = nextCenter;
            lastIndex = index;

            if (index < _fiberIds.size() - 1) {
                ++index;
            }
            nextCenter = getXCenter(_fiberIds[index], iy);
        }

        _xToFiberId[x] = _fiberIds[lastIndex];
    }
}           

}}}
