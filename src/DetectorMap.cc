#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/daf/base.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {

/*
 * ctor
 */
DetectorMap::DetectorMap(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds, // 1-indexed IDs for each fibre
                         ndarray::Array<float, 2, 1> const& xCenters,    // center of trace for each fibre
                         ndarray::Array<float, 2, 1> const& wavelengths, // wavelengths for each fibre
                         std::size_t nKnot,                               // number of knots
                         ndarray::Array<float, 2, 1> const& slitOffsets, // per-fibre x, y, focus offsets
                         ndarray::Array<float, 1, 1> const& throughput
                        ) :
    _nFiber(fiberIds.getShape()[0]),
    _bbox(bbox),
    _fiberIds(ndarray::copy(fiberIds)),
    _throughput(_nFiber),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _nKnot(nKnot),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(ndarray::makeVector(std::size_t(3), _nFiber)),
    _visitInfo(lsst::daf::base::PropertyList()),
    _metadata(std::make_shared<lsst::daf::base::PropertyList>())
{
    /*
     * Check inputs
     */
    utils::checkSize(wavelengths.getShape(), xCenters.getShape(), "DetectorMap: wavelength vs xCenters");
    utils::checkSize(_fiberIds.getNumElements(), xCenters.getShape()[0], "DetectorMap: fiberIds vs xCenters");
    utils::checkSize(std::size_t(_bbox.getHeight()), xCenters.getShape()[1], "DetectorMap: bbox vs xCenters");

    if (!slitOffsets.isEmpty()) {
        setSlitOffsets(slitOffsets);   // actually this is where we check slitOffsets
    } else {
        _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
    }

    if (throughput.isEmpty()) {
        _throughput.deep() = 1.0;
    } else {
        setThroughput(throughput);
    }

    _setSplines(xCenters, wavelengths);
}


DetectorMap::DetectorMap(
    lsst::afw::geom::Box2I bbox,
    FiberMap const& fiberIds,
    ndarray::Array<float const, 2, 1> const& centerKnots,
    ndarray::Array<float const, 2, 1> const& centerValues,
    ndarray::Array<float const, 2, 1> const& wavelengthKnots,
    ndarray::Array<float const, 2, 1> const& wavelengthValues,
    Array2D const& slitOffsets,
    Array1D const& throughput,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _nFiber(fiberIds.getShape()[0]),
    _bbox(bbox),
    _fiberIds(ndarray::copy(fiberIds)),
    _throughput(throughput),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _nKnot(centerKnots.getShape()[1]),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(slitOffsets),
    _visitInfo(visitInfo),
    _metadata(metadata)
{
    utils::checkSize(centerKnots.getShape()[0], _nFiber, "DetectorMap: nFiber");
    utils::checkSize(centerKnots.getShape(), centerValues.getShape(),
                     "DetectorMap: centerKnots vs centerValues");
    utils::checkSize(wavelengthKnots.getShape(), wavelengthValues.getShape(),
                     "DetectorMap:: wavelengthKnots vs wavelengthValues");
    utils::checkSize(centerKnots.getShape(), wavelengthKnots.getShape(),
                     "DetectorMap:: centerKnots vs wavelengthKnots");

    for (std::size_t ii = 0; ii < _nFiber; ++ii) {
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

DetectorMap::Array2D DetectorMap::getWavelength() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    Array2D result{numFibers, std::size_t(_bbox.getHeight())};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii].deep() = getWavelength(_fiberIds[ii]);
    }
    return result;
}

/*
 * Return the wavelength values for a fibre
 */
void
DetectorMap::setWavelength(std::size_t fiberId,
                           ndarray::Array<float, 1, 1> const& wavelength)
{
    int const index = getFiberIndex(fiberId);
    auto const xc = wavelength;         // not really updated, but I need an array

    _setSplines(index, xc, false, wavelength, true);
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

DetectorMap::Array2D DetectorMap::getXCenter() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    Array2D result{numFibers, std::size_t(_bbox.getHeight())};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii].deep() = getXCenter(_fiberIds[ii]);
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

/*
 * Update the xcenter values for a fibre
 */
void
DetectorMap::setXCenter(std::size_t fiberId,
                        ndarray::Array<float, 1, 1> const& xCenter)
{
    int const index = getFiberIndex(fiberId);
    auto const wavelength = xCenter;    // not really updated, but I need an array

    _setSplines(index, xCenter, true, wavelength, false);
}

/*
 * Return a fibre's throughput
 */
float
DetectorMap::getThroughput(std::size_t fiberId) const
{
    int const index = getFiberIndex(fiberId);

    return _throughput[index];
}

/*
 * Return a fibre's throughput
 */
void
DetectorMap::setThroughput(std::size_t fiberId,
                           float throughput)
{
    int const index = getFiberIndex(fiberId);

    _throughput[index] = throughput;
}

void DetectorMap::setThroughput(Array1D const& throughput) {
    if (throughput.getShape() != _throughput.getShape()) {
        std::ostringstream os;
        os << "Length mismatch: " << throughput.getShape() << " vs " << _throughput.getShape();
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
    _throughput.deep() = throughput;
}

/*
 * Return the position of the fiber trace on the detector, given a fiberId and wavelength
 */
lsst::afw::geom::PointD
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
        return lsst::afw::geom::PointD(NaN, NaN);
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

    return lsst::afw::geom::PointD(x, y);
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

    return spline(row);
}

/************************************************************************************************************/
/*
 * Return the fiberId given a position on the detector
 */
int
DetectorMap::findFiberId(lsst::afw::geom::PointD pixelPos // position on detector
                          ) const
{
    float const x = pixelPos[0], y = pixelPos[1];
    std::size_t const maxIter = 2*_fiberIds.getNumElements();  // maximum number of iterations

    if (!_bbox.contains(lsst::afw::geom::PointI(x, y))) {
        std::ostringstream os;
        os << "Point " << pixelPos << " does not lie within BBox " << _bbox;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }

    int fiberId = _xToFiberId[int(x)];    // first guess at the fiberId
    std::size_t index = getFiberIndex(fiberId);
    std::size_t iterations = 0;
    for (bool updatedIndex = true; updatedIndex; ++iterations) {
        auto const & spline0 = getCenterSpline(index);
        float xCenter0 = spline0(y);

        updatedIndex = false;
        if (index > 0) {
            auto const & splineMinus = getCenterSpline(index - 1);
            float const xCenterMinus = splineMinus(y);

            if (std::fabs(x - xCenterMinus) < std::fabs(x - xCenter0)) {
                --index;
                updatedIndex = true;
                continue;
            }
        }
        if (index < _yToXCenter.size() - 1) {
            auto const & splinePlus = getCenterSpline(index + 1);
            float const xCenterPlus = splinePlus(y);

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

void
DetectorMap::_setSplines(ndarray::Array<float, 2, 1> const& xCenters,     // center of trace for each fibre
                         ndarray::Array<float, 2, 1> const& wavelengths // wavelengths for each fibre
                        )
{
    // values of y used as domain for the spline
    std::vector<float> yIndices(_nKnot);
    // Two vectors with the values for a single fibre
    std::vector<float> xCenter(_nKnot);
    std::vector<float> wavelength(_nKnot);
    /*
     * loop over the fibers, setting the splines.  Note the index != fiberId (that's _fiberId[index])
     */
    for (std::size_t index = 0; index != _nFiber; ++index) {
        _setSplines(index, xCenters[index], true, wavelengths[index], true);
    }

    _set_xToFiberId();
}

/*
 * set _xToFiberId, an array giving the fiber ID for each pixel across the centre of the chip
 */
void
DetectorMap::_set_xToFiberId()
{
    int const iy = _bbox.getHeight()/2;

    float lastCenter = -1e30;            // center of the last fibre we passed scanning to the right
    std::size_t lastIndex = -1;          // index of the last fibre we passed.  Special case starting with 0

    float nextCenter = getXCenter(_fiberIds[1], iy); // center of the next fiber we're looking at
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

void
DetectorMap::_setSplines(std::size_t index,               // desired fiducial index
                         ndarray::Array<float, 1, 1> const& xc, // center of trace for each fibre
                         bool setXCenters,                      // set the xCenter values?
                         ndarray::Array<float, 1, 1> const& wl, // wavelengths for each fibre
                         bool setWavelengths                    // set the wavelength values?
                        )
{
    // values of y used as domain for the spline
    ndarray::Array<float, 1, 1> yIndices = ndarray::allocate(_nKnot);
    // Two vectors with the values for a single fibre
    ndarray::Array<float, 1, 1> xCenter = ndarray::allocate(_nKnot);
    ndarray::Array<float, 1, 1> wavelength = ndarray::allocate(_nKnot);
    /*
     * Setting the splines
     */
    // look for finite values
    std::size_t j;
    for (j = 0; j != xc.size(); ++j) {
        if (std::isfinite(wl[j] + xc[j])) {
            break;
        }
    }
    std::size_t const j0 = j;

    for (j = xc.size() - 1; j >= 0; j--) {
        if (std::isfinite(wl[j] + xc[j])) {
            break;
        }
    }
    std::size_t const j1 = j - 1;
    /*
     * OK, we know that we have finite values from j0..j1, so construct the vectors
     */
    float const dy = (j1 - j0 + 1.0)/(_nKnot - 1); // step in y
    float y = j0;
    for (std::size_t i = 0; i != _nKnot; ++i, y += dy) {
        int const iy = std::floor(y);
        
        yIndices[i] = iy;
        xCenter[i]    = xc[iy];
        wavelength[i] = wl[iy];
    }
    /*
     * We have the arrays so we can set up the splines for the fibre
     */
    if (setXCenters) {
        _yToXCenter[index] = std::make_shared<math::Spline<float>>(yIndices, xCenter);
    }
    if (setWavelengths) {
        _yToWavelength[index] = std::make_shared<math::Spline<float>>(yIndices, wavelength);
    }
}
}}}
