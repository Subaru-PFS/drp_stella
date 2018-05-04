#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "lsst/pex/exceptions/Exception.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {

const int DetectorMap::FIBER_DX;
const int DetectorMap::FIBER_DY;
const int DetectorMap::FIBER_DFOCUS;

/*
 * ctor
 */
DetectorMap::DetectorMap(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds,      // 1-indexed IDs for each fibre
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
    _slitOffsets(ndarray::makeVector(3, _nFiber))
{
    /*
     * Check inputs
     */
    if (wavelengths.getShape() != xCenters.getShape()) {
        std::ostringstream os;
        os << "Shape of wavelengths == " << wavelengths.getShape() <<
            " != xCenters == " << xCenters.getShape();
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }

    if (_fiberIds.getNumElements() != xCenters.getShape()[0]) {
        std::ostringstream os;
        os << "Number of wavelengths/xCenters arrays == " << xCenters.getShape()[0] <<
            " != nFiber == " << _fiberIds.size();
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }

    if (_bbox.getHeight() != xCenters.getShape()[1]) {
        std::ostringstream os;
        os << "Length of wavelengths/xCenters arrays == " << xCenters.getShape()[1] <<
            " != height of bbox == " << _bbox.getHeight();
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
    /*
     * OK; arguments are checked
     */
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

/*
 * ctor.
 *
 * N.b. this is protected, and is for the use of DetectorMapIO only.  The DetectorMap
 * created by this ctor isn't really ready for use
 */
DetectorMap::DetectorMap(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                         FiberMap const& fiberIds,      // 1-indexed IDs for each fibre
                         std::size_t nKnot                               // number of knots
                        ) :
    _nFiber(fiberIds.getShape()[0]),
    _bbox(bbox),
    _fiberIds(ndarray::copy(fiberIds)),
    _throughput(_nFiber, 1.0),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _nKnot(nKnot),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(ndarray::makeVector(3, _nFiber))
{
    _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
}
            
/*
 * Return a fiberIdx given a fiberId
 */
std::size_t
DetectorMap::getFiberIdx(std::size_t fiberId) const
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

/*
 * Return the wavelength values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getWavelength(std::size_t fiberId) const
{
    int const fidx = getFiberIdx(fiberId);
    auto const & spline = _yToWavelength[fidx];
    const float slitOffsetY = _slitOffsets[FIBER_DY][fidx];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY);
    }

    return res;
}

DetectorMap::Array2D DetectorMap::getWavelength() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    Array2D result{numFibers, static_cast<std::size_t>(_bbox.getHeight())};
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
    int const fidx = getFiberIdx(fiberId);
    auto const xc = wavelength;         // not really updated, but I need an array

    _setSplines(fidx, xc, false, wavelength, true);
}
            
/*
 * Return the xCenter values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getXCenter(std::size_t fiberId) const
{
    int const fidx = getFiberIdx(fiberId);
    auto const & spline = _yToXCenter[fidx];
    const float slitOffsetX = _slitOffsets[FIBER_DX][fidx];
    const float slitOffsetY = _slitOffsets[FIBER_DY][fidx];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY) + slitOffsetX;
    }

    return res;
}

DetectorMap::Array2D DetectorMap::getXCenter() const {
    std::size_t numFibers = _fiberIds.getNumElements();
    Array2D result{numFibers, static_cast<std::size_t>(_bbox.getHeight())};
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
    int const fidx = getFiberIdx(fiberId);
    auto const & spline = _yToXCenter[fidx];
    const float slitOffsetX = _slitOffsets[FIBER_DX][fidx];
    const float slitOffsetY = _slitOffsets[FIBER_DY][fidx];

    return spline(y - slitOffsetY) + slitOffsetX;
}

/*
 * Update the xcenter values for a fibre
 */
void
DetectorMap::setXCenter(std::size_t fiberId,
                        ndarray::Array<float, 1, 1> const& xCenter)
{
    int const fidx = getFiberIdx(fiberId);
    auto const wavelength = xCenter;    // not really updated, but I need an array

    _setSplines(fidx, xCenter, true, wavelength, false);
}

/*
 * Return a fibre's throughput
 */
float
DetectorMap::getThroughput(std::size_t fiberId) const
{
    int const fidx = getFiberIdx(fiberId);

    return _throughput[fidx];
}

/*
 * Return a fibre's throughput
 */
void
DetectorMap::setThroughput(std::size_t fiberId,
                           const float throughput)
{
    int const fidx = getFiberIdx(fiberId);

    _throughput[fidx] = throughput;
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
DetectorMap::findPoint(const int fiberId,               ///< Desired fibreId
                       const float wavelength           ///< desired wavelength
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
DetectorMap::findWavelength(const int fiberId,               ///< Desired fibreId
                            const float row                  ///< desired row
                           ) const
{
    if (row < 0 || row > _bbox.getHeight() - 1) {
        std::ostringstream os;
        os << "Row " << row << " is not in range [0, " << _bbox.getHeight() - 1 << "]";
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    int const fidx = getFiberIdx(fiberId);
    auto const & spline = _yToWavelength[fidx];

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

    if (!_bbox.contains(lsst::afw::geom::PointI(x, y))) {
        std::ostringstream os;
        os << "Point " << pixelPos << " does not lie within BBox " << _bbox;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }

    int fiberId = _xToFiberId[(int)(x)];    // first guess at the fiberId
    int fidx = getFiberIdx(fiberId);
    bool updatedFidx = true;
    while (updatedFidx) {
        auto const & spline_0 = _yToXCenter[fidx];
        float xCenter_0 = spline_0(y);

        updatedFidx = false;
        if (fidx > 0) {
            auto const & spline_m1 = _yToXCenter[fidx - 1];
            float xCenter_m1 = spline_m1(y);

            if (std::fabs(x - xCenter_m1) < std::fabs(x - xCenter_0)) {
                fidx--;
                updatedFidx = true;
            }
        }
        if (fidx < _yToXCenter.size() - 1) {
            auto const & spline_p1 = _yToXCenter[fidx + 1];
            float xCenter_p1 = spline_p1(y);

            if (std::fabs(x - xCenter_p1) < std::fabs(x - xCenter_0)) {
                fidx++;
                updatedFidx = true;
            }
        }
    }

    return _fiberIds[fidx];
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
     * loop over the fibers, setting the splines.  Note the fidx != fiberId (that's _fiberId[idx])
     */
    for (int fidx = 0; fidx != _nFiber; ++fidx) {
        _setSplines(fidx, xCenters[fidx], true, wavelengths[fidx], true);
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

    float last_xc = -1e30;              // center of the last fibre we passed scanning to the right
    int last_fidx = -1;                  // fidx of the last fibre we passed.  Special case starting with 0

    float next_xc = getXCenter(_fiberIds[1], iy); // center of the next fiber we're looking at
    for (int fidx = 0, x = 0; x != _bbox.getWidth(); ++x) {
        if (x - last_xc > next_xc - x) { // x is nearer to next_xc than last_xc
            last_xc = next_xc;
            last_fidx = fidx;

            if (fidx < _fiberIds.size() - 1) {
                ++fidx;
            }
            next_xc = getXCenter(_fiberIds[fidx], iy);
        }

        _xToFiberId[x] = _fiberIds[last_fidx];
    }
}           

void
DetectorMap::_setSplines(const std::size_t fidx,                // desired fiducial index
                         ndarray::Array<float, 1, 1> const& xc, // center of trace for each fibre
                         bool setXCenters,                      // set the xCenter values?
                         ndarray::Array<float, 1, 1> const& wl, // wavelengths for each fibre
                         bool setWavelengths                    // set the wavelength values?
                        )
{
    // values of y used as domain for the spline
    std::vector<float> yIndices(_nKnot);
    // Two vectors with the values for a single fibre
    std::vector<float> xCenter(_nKnot);
    std::vector<float> wavelength(_nKnot);    
    /*
     * Setting the splines
     */
    // look for finite values
    int j;
    for (j = 0; j != xc.size(); ++j) {
        if (std::isfinite(wl[j] + xc[j])) {
            break;
        }
    }
    int const j0 = j;

    for (j = xc.size() - 1; j >= 0; j--) {
        if (std::isfinite(wl[j] + xc[j])) {
            break;
        }
    }
    int const j1 = j - 1;
    /*
     * OK, we know that we have finite values from j0..j1, so construct the vectors
     */
    float const dy = (j1 - j0 + 1.0)/(_nKnot - 1); // step in y
    float y = j0;
    for (int i = 0; i != _nKnot; ++i, y += dy) {
        int const iy = std::floor(y);
        
        yIndices[i] = iy;
        xCenter[i]    = xc[iy];
        wavelength[i] = wl[iy];
    }
    /*
     * We have the arrays so we can set up the splines for the fibre
     */
    if (setXCenters) {
        _yToXCenter[fidx] = math::spline<float>(yIndices, xCenter);       // first xCenter
    }
    if (setWavelengths) {
        _yToWavelength[fidx] = math::spline<float>(yIndices, wavelength); // then wavelength
    }
}
}}}
