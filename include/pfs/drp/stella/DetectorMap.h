#if !defined(PFS_DRP_STELLA_DETECTORMAP_H)
#define PFS_DRP_STELLA_DETECTORMAP_H

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
    friend class DetectorMapIO;
public:
    static const int FIBER_DX=0, FIBER_DY=1, FIBER_DFOCUS=2;

    /** \brief ctor */
    explicit DetectorMap(lsst::afw::geom::Box2I bbox,                    ///< detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds,      ///< 1-indexed IDs for each fibre
                         ndarray::Array<float, 2, 1> const& xCenters,    ///< center of trace for each fibre
                         ndarray::Array<float, 2, 1> const& wavelengths, ///< wavelengths for each fibre
                         ndarray::Array<float, 2, 1> const* slitOffsets, ///< per-fibre offsets
                         std::size_t nKnot                               ///< number of knots to use
                        );

protected:
    explicit DetectorMap(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                         ndarray::Array<int, 1, 1> const& fiberIds,      // 1-indexed IDs for each fibre
                         std::size_t nKnot                               // number of knots
                        );
public:    
    /** \brief dtor */
    virtual ~DetectorMap() {}

    /** \brief return the bbox */
    lsst::afw::geom::Box2I getBBox() const { return _bbox; }

    /** \brief return the fiberIds */
    std::vector<int> & getFiberIds() { return _fiberIds; }

    /**
     * Set the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre.
     *
     * See getSlitOffsets()
     */
    void setSlitOffsets(ndarray::Array<float, 2, 1> const& slitOffsets ///< new values of offsets
                       );

    /**
     * Set the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * for a fibre.
     *
     * See getSlitOffsets()
     */
    void setSlitOffsets(std::size_t fiberId, ///< desired fibre
                        ndarray::Array<float, 1, 0> const& offsets ///< slit offsets for chosen fibre
                       )
        {
            _slitOffsets[ndarray::view(FIBER_DX, FIBER_DFOCUS + 1)(getFiberIdx(fiberId))].deep() = offsets;
        }
    
    /**
     * Get the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre
     *
     * The return value is a (3, nFiber) array, which may be indexed by FIBER_DX, FIBER_DY, and FIBER_DFOCUS,
     * e.g.
     *  ftMap.getSlitOffsets()[FIBER_DX][100]
     * is the offset in the x-direction for the 100th fiber (n.b. not fiberId necessarily; cf. findFiberId())
     *
     * The units are pixels for DX and DY; microns at the slit for focus
     */
    ndarray::Array<float, 2, 1> const& getSlitOffsets() const { return _slitOffsets; }
    /**
     * Return the slit offsets for the specified fibre; see getSlitOffsets() for details
     * e.g.
     *  ftMap.getSlitOffsets(fiberId)[FIBER_DY]
     * is the offset in the y-direction for the fibre identified by fiberId
     */
    ndarray::Array<float, 1, 0> const getSlitOffsets(std::size_t fiberId ///< fiberId
                                                    ) const
        { return _slitOffsets[ndarray::view(FIBER_DX, FIBER_DFOCUS + 1)(getFiberIdx(fiberId))]; }

    /**
     * Return the wavelength values for a fibre
     */
    ndarray::Array<float, 1, 1> getWavelength(std::size_t fiberId ///< fiberId
                                             ) const;

    /**
     * Set the wavelength values for a fibre
     */
    void setWavelength(std::size_t fiberId,                       ///< 1-indexed fiberID 
                       ndarray::Array<float, 1, 1> const& wavelength ///< wavelengths for fibre
                      );

    /**
     * Return the xCenter values for a fibre
     */
    ndarray::Array<float, 1, 1> getXCenter(std::size_t fiberId ///< fiberId
                                          ) const;
protected:
    float getXCenter(std::size_t fiberId, ///< fiberId
                     float y              ///< desired y value
                    ) const;
public:

    /**
     * Set the xCenter values for a fibre
     */
    void setXCenter(std::size_t fiberId,                       ///< 1-indexed fiberID 
                    ndarray::Array<float, 1, 1> const& xCenter ///< center of trace for fibre
                   );

    /** \brief
     * Return the (relative) throughput for a fibre
     */
    float getThroughput(std::size_t fiberId                       ///< 1-indexed fiberID 
                       ) const;

    /** \brief
     * Set a fibre's (relative) throughput
     */
    void setThroughput(std::size_t fiberId,                       ///< 1-indexed fiberID 
                       const float throughput                     ///< the fibre's throughput
                      );

    /** \brief
     * Return the fiberId given a position on the detector
     */
    int findFiberId(lsst::afw::geom::PointD pixelPos ///< position on detector
                   ) const;

    /** \brief
     * Return the position of the fiber trace on the detector, given a fiberId and wavelength
     */
    lsst::afw::geom::PointD findPoint(const int fiberId,               ///< Desired fibreId
                                      const float wavelength           ///< desired wavelength
                                     ) const;
    /** \brief
     * Return the wavelength of a point on the detector, given a fiberId and position
     */
    float findWavelength(const int fiberId,               ///< Desired fibreId
                         const float pixelPos             ///< desired row
                        ) const;
    /** \brief
     * Return the index of a fiber, given its fiber ID
     */
    std::size_t getFiberIdx(std::size_t fiberId) const;
private:                                // initialise before _yTo{XCenter,Wavelength}
    int _nFiber;                        // number of fibers
protected:
    // N.b. DetectorMapIO is a friend, and makes the protected members available for read/write routines
    lsst::afw::geom::Box2I _bbox;       // bounding box of detector
    std::vector<int> _fiberIds;         // The fiberIds (between 1 and c. 2400) present on this detector
    
    std::vector<float> _throughput;	// The throughput (in arbitrary units ~ 1) of each fibre
    //
    // These std::vectors are indexed by fiberIdx (not fiberId)
    //
    std::vector<math::spline<float>> _yToXCenter; // splines to convert a y pixel value to trace position
    std::vector<math::spline<float>> _yToWavelength; // splines to convert a y pixel value to wavelength

    void _set_xToFiberId();
private:
    int _nKnot;                         // number of knots for splines
    //
    // An array that gives the fiberId half way up the chip
    //
    ndarray::Array<int, 1, 1> _xToFiberId;
    //
    // offset (in pixels) for each trace in x, and y and in focus (microns at the slit); indexed by fiberIdx
    //
    ndarray::Array<float, 2, 1> _slitOffsets; 
    /*
     * Private helper functions
     */
    void _setSplines(const std::size_t fidx,
                     ndarray::Array<float, 1, 1> const& xc, bool setXCenters,
                     ndarray::Array<float, 1, 1> const& wl, bool setWavelengths);

    void _setSplines(ndarray::Array<float, 2, 1> const&,
                     ndarray::Array<float, 2, 1> const&);
};

}}}

#endif
