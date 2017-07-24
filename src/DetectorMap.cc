#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "lsst/pex/exceptions/Exception.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {
/*
 * ctor
 */
DetectorMap::DetectorMap(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds,      // 1-indexed IDs for each fibre
                         ndarray::Array<float, 2, 1> const& xCenters,    // center of trace for each fibre
                         ndarray::Array<float, 2, 1> const& wavelengths, // wavelengths for each fibre
                         ndarray::Array<float, 1, 1> const* slitOffsets, // per-fibre wavelength offsets
                         std::size_t nKnot                               // number of knots
                        ) :
    _nFiber(fiberIds.getShape()[0]),
    _bbox(bbox),
    _fiberIds(fiberIds.begin(), fiberIds.end()),
    _nKnot(nKnot),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(_nFiber)
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

    if (_fiberIds.size() != xCenters.getShape()[0]) {
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
    if (slitOffsets) {
        setSlitOffsets(*slitOffsets);   // actually this is where we check slitOffsets
    } else {
        _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
    }

    _setSplines(xCenters, wavelengths);
}

/*
 * Return a fiberIdx given a fiberId
 */
std::size_t
DetectorMap::_getFiberIdx(std::size_t fiberId) const
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
DetectorMap::setSlitOffsets(ndarray::Array<float, 1, 1> const& slitOffsets)
{
    if (slitOffsets.getShape()[0] != _nFiber) {
        std::ostringstream os;
        os << "Number of offsets == " << slitOffsets.getShape()[0] <<
            " != number of fibres == " << _nFiber;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }

    _slitOffsets = slitOffsets;
}
            
/*
 * Return the wavelength or xCenter values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::_getSomething(std::vector<math::spline<float>> const& something,
                           std::size_t fiberId,
                           bool const applySlitOffset
                          ) const
{
    int const fidx = _getFiberIdx(fiberId);
    auto const & spline = something[fidx];
    float slitOffset = applySlitOffset ? _slitOffsets[fidx] : 0.0;

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffset);
    }

    return res;
}

/*
 * Return the wavelength values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getWavelength(std::size_t fiberId) const
{
    return _getSomething(_yToWavelength, fiberId, true);
}

/*
 * Return the xCenter values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getXCenter(std::size_t fiberId) const
{
    return _getSomething(_yToXCenter, fiberId, false);
}
            
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
    int fidx = _getFiberIdx(fiberId);
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
    math::spline<float> spline;
    for (int fidx = 0; fidx != _nFiber; ++fidx) {
        // look for finite values
        auto const wl = wavelengths[fidx];
        auto const xc = xCenters[fidx];
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
         * OK, we know that have finite values from j0..j1, so construct the vectors         * 
         */
        float const dy = (j1 - j0 + 1.0)/(_nKnot - 1); // step in y
        float y = j0;
        for (int i = 0; i != _nKnot; ++i, y += dy) {
            int const iy = std::floor(y);

            yIndices[i] = iy;
            xCenter[i]    = xCenters[fidx][iy];
            wavelength[i] = wavelengths[fidx][iy];
        }
        /*
         * We have the arrays so we can set up the splines for each fibre
         */
        spline.set_points(yIndices, xCenter); // first xCenter
        _yToXCenter[fidx] = spline;           // N.b. moves/copies spline into the vector

        spline.set_points(yIndices, wavelength); // then wavelength
        _yToWavelength[fidx] = spline;           // N.b. moves/copies spline into the vector
    }
    /*
     * Now set _xToFiberId, an array giving the fiber ID for each pixel across the centre of the chip
     */
    int const iy = _bbox.getHeight()/2;

    float last_xc = -1e30;              // center of the last fibre we passed scanning to the right
    int last_fidx = -1;                  // fidx of the last fibre we passed.  Special case starting with 0

    float next_xc = xCenters[1][iy]; // center of the next fiber we're looking at
    for (int fidx = 0, x = 0; x != _bbox.getWidth(); ++x) {
        if (x - last_xc > next_xc - x) { // x is nearer to next_xc than last_xc
            last_xc = next_xc;
            last_fidx = fidx;

            if (fidx < _fiberIds.size() - 1) {
                ++fidx;
            }
            next_xc = xCenters[fidx][iy];
        }

        _xToFiberId[x] = _fiberIds[last_fidx];
    }
}
}}}
