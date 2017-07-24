#if !defined(PFS_DRP_STELLA_FIBERTRACEMAP_H)
#define PFS_DRP_STELLA_FIBERTRACEMAP_H

#include <vector>
#include "ndarray.h"

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"

#include "pfs/drp/stella/spline.h"

namespace pfs { namespace drp { namespace stella {
/**
 * @brief Describe the geometry of the focal plane.
 *
 * The information provided is the mapping of the position and wavelength of the fiber traces onto
 * the CCD.
 *
 * The slitOffsets are the different zeropoints of the wavelengths of each fibre (in floating-point pixels),
 * due to imperfect manufacture of the slithead.  The values are *subtracted* from the CCD coordinates
 * when calculating the wavelength for a pixel -- i.e. we assume that the spots are deflected up by slitOffset
 */
class DetectorMap {
public:
    /** \brief ctor */
    explicit DetectorMap(lsst::afw::geom::Box2I bbox,                    ///< detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds,      ///< 1-indexed IDs for each fibre
                         ndarray::Array<float, 2, 1> const& xCenters,    ///< center of trace for each fibre
                         ndarray::Array<float, 2, 1> const& wavelengths, ///< wavelengths for each fibre
                         ndarray::Array<float, 1, 1> const* slitOffsets, ///< per-fibre wavelength offsets
                         std::size_t nKnot                               ///< number of knots to use
                        );

    /** \brief dtor */
    virtual ~DetectorMap() {}

    /** \brief return the fiberIds */
    std::vector<int> & getFiberIds() { return _fiberIds; }

    /**
     * Set the offsets of the wavelengths of each fibre (in floating-point pixels)
     */
    void setSlitOffsets(ndarray::Array<float, 1, 1> const& slitOffsets ///< new values of offsets
                       );

    /**
     * Return the offsets of the wavelengths of each fibre (in floating-point pixels)
     */
    ndarray::Array<float, 1, 1> const& getSlitOffsets() const { return _slitOffsets; }

    /**
     * Return the wavelength values for a fibre
     */
    ndarray::Array<float, 1, 1> getWavelength(std::size_t fiberId ///< fiberId
                                             ) const;

    /**
     * Return the xCenter values for a fibre
     */
    ndarray::Array<float, 1, 1> getXCenter(std::size_t fiberId ///< fiberId
                                          ) const;

    /** \brief
     * Return the fiberId given a position on the detector
     */
    int findFiberId(lsst::afw::geom::PointD pixelPos ///< position on detector
                   ) const;
private:
    int _nFiber;                        // number of fibers
    lsst::afw::geom::Box2I _bbox;       // bounding box of detector
    std::vector<int> _fiberIds;         // The 1-indexed fiberIds present on this detector
    //
    // These std::vectors are indexed by fiberID (which start at 1)
    //
    int _nKnot;                         // number of knots for splines
    std::vector<math::spline<float>> _yToXCenter; // splines to convert a y pixel value to trace position
    std::vector<math::spline<float>> _yToWavelength; // splines to convert a y pixel value to wavelength
    //
    // An array that gives the fiberId half way up the chip
    //
    ndarray::Array<int, 1, 1> _xToFiberId;
    //
    // offset (in pixels) for each trace in lambda direction, indexed by fiberId
    //
    ndarray::Array<float, 1, 1> _slitOffsets; 
    /*
     * Private helper functions
     */
    std::size_t _getFiberIdx(std::size_t fiberId) const;

    ndarray::Array<float, 1, 1> _getSomething(std::vector<math::spline<float>> const&,
                                              std::size_t, bool const) const;

    void _setSplines(ndarray::Array<float, 2, 1> const&,
                     ndarray::Array<float, 2, 1> const&);
};

}}}

#endif
