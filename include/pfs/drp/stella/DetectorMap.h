#if !defined(PFS_DRP_STELLA_DETECTORMAP_H)
#define PFS_DRP_STELLA_DETECTORMAP_H

#include <vector>
#include "ndarray.h"

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/VisitInfo.h"

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
    enum ArrayRow { DX = 0, DY = 1, DFOCUS = 2 };

    using FiberMap = ndarray::Array<int, 1, 1>;
    using Array2D = ndarray::Array<float, 2, 1>;
    using Array1D = ndarray::Array<float, 1, 1>;

    /** \brief ctor */
    explicit DetectorMap(lsst::afw::geom::Box2I bbox, ///< detector's bounding box
                         FiberMap const& fiberIds, ///< 1-indexed IDs for each fibre
                         Array2D const& xCenters,  ///< center of trace for each fibre
                         Array2D const& wavelengths, ///< wavelengths for each fibre
                         std::size_t nKnot, ///< number of knots to use
                         Array2D const& slitOffsets=Array2D(), ///< per-fibre offsets
                         Array1D const& throughput=Array1D() ///< relative throughput per fiber
                        );

    DetectorMap(lsst::afw::geom::Box2I bbox,  // detector's bounding box
                FiberMap const& fiberIds,  // 1-indexed IDs for each fibre
                ndarray::Array<float const, 2, 1> const& centerKnots,
                ndarray::Array<float const, 2, 1> const& centerValues,
                ndarray::Array<float const, 2, 1> const& wavelengthKnots,
                ndarray::Array<float const, 2, 1> const& wavelengthValues,
                Array2D const& slitOffsets,  // per-fibre offsets
                Array1D const& throughput  // relative throughput per fiber
                );

    virtual ~DetectorMap() {}
    DetectorMap(DetectorMap const&) = default;
    DetectorMap(DetectorMap &&) = default;
    DetectorMap & operator=(DetectorMap const&) = default;
    DetectorMap & operator=(DetectorMap &&) = default;

    /** \brief return the bbox */
    lsst::afw::geom::Box2I getBBox() const { return _bbox; }

    int getNKnot() const { return _nKnot; }

    /** \brief return the fiberIds */
    FiberMap & getFiberIds() { return _fiberIds; }
    FiberMap const& getFiberIds() const { return _fiberIds; }

    /**
     * Set the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre.
     *
     * See getSlitOffsets()
     */
    void setSlitOffsets(Array2D const& slitOffsets ///< new values of offsets
                       );

    /**
     * Set the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * for a fibre.
     *
     * See getSlitOffsets()
     */
    void setSlitOffsets(
        std::size_t fiberId, ///< desired fibre
        ndarray::Array<float, 1, 0> const& offsets ///< slit offsets for chosen fibre
    ) {
        _slitOffsets[ndarray::view(DX, DFOCUS + 1)(getFiberIndex(fiberId))].deep() = offsets;
    }
    
    /**
     * Get the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre
     *
     * The return value is a (3, nFiber) array, which may be indexed by DX, DY, and DFOCUS,
     * e.g.
     *  ftMap.getSlitOffsets()[DX][100]
     * is the offset in the x-direction for the 100th fiber (n.b. not fiberId necessarily; cf. findFiberId())
     *
     * The units are pixels for DX and DY; microns at the slit for focus
     */
    Array2D const& getSlitOffsets() const { return _slitOffsets; }
    /**
     * Return the slit offsets for the specified fibre; see getSlitOffsets() for details
     * e.g.
     *  ftMap.getSlitOffsets(fiberId)[DY]
     * is the offset in the y-direction for the fibre identified by fiberId
     */
    ndarray::Array<float const, 1, 0> const getSlitOffsets(
        std::size_t fiberId ///< fiberId
    ) const { return _slitOffsets[ndarray::view(DX, DFOCUS + 1)(getFiberIndex(fiberId))]; }

    /**
     * Return the wavelength values for a fibre
     */
    Array1D getWavelength(std::size_t fiberId ///< fiberId
                                             ) const;
    Array2D getWavelength() const;

    /**
     * Set the wavelength values for a fibre
     */
    void setWavelength(std::size_t fiberId,                       ///< 1-indexed fiberID 
                       Array1D const& wavelength ///< wavelengths for fibre
                      );

    //@{
    /**
     * Return the xCenter values for a fibre
     */
    Array1D getXCenter(std::size_t fiberId ///< fiberId
                                          ) const;
    Array2D getXCenter() const;
protected:
    float getXCenter(std::size_t fiberId, ///< fiberId
                     float y              ///< desired y value
                     ) const;
    //@}

public:

    /**
     * Set the xCenter values for a fibre
     */
    void setXCenter(std::size_t fiberId,                       ///< 1-indexed fiberID 
                    Array1D const& xCenter ///< center of trace for fibre
                   );

    /** \brief
     * Return the (relative) throughput for a fibre
     */
    float getThroughput(std::size_t fiberId                       ///< 1-indexed fiberID 
                       ) const;

    Array1D & getThroughput() { return _throughput; }
    Array1D const& getThroughput() const { return _throughput; }

    /** \brief
     * Set a fibre's (relative) throughput
     */
    void setThroughput(std::size_t fiberId,                       ///< 1-indexed fiberID 
                       float throughput                     ///< the fibre's throughput
                      );

    void setThroughput(Array1D const& throughput);

    /** \brief
     * Return the fiberId given a position on the detector
     */
    int findFiberId(lsst::afw::geom::PointD pixelPos ///< position on detector
                   ) const;

    /** \brief
     * Return the position of the fiber trace on the detector, given a fiberId and wavelength
     */
    lsst::afw::geom::PointD findPoint(int fiberId,               ///< Desired fibreId
                                      float wavelength           ///< desired wavelength
                                     ) const;
    /** \brief
     * Return the wavelength of a point on the detector, given a fiberId and position
     */
    float findWavelength(int fiberId,               ///< Desired fibreId
                         float pixelPos             ///< desired row
                        ) const;
    /** \brief
     * Return the index of a fiber, given its fiber ID
     */
    std::size_t getFiberIndex(int fiberId) const;

    lsst::afw::image::VisitInfo getVisitInfo() { return _visitInfo; }
    void setVisitInfo(lsst::afw::image::VisitInfo &visitInfo) { _visitInfo = visitInfo; };

    math::Spline<float> const& getCenterSpline(std::size_t index) const;
    math::Spline<float> const& getWavelengthSpline(std::size_t index) const;

private:                                // initialise before _yTo{XCenter,Wavelength}
    std::size_t _nFiber;                // number of fibers
    lsst::afw::geom::Box2I _bbox;       // bounding box of detector
    FiberMap _fiberIds;         // The fiberIds (between 1 and c. 2400) present on this detector
    
    Array1D _throughput;	// The throughput (in arbitrary units ~ 1) of each fibre
    //
    // These std::vectors are indexed by fiberIndex (not fiberId)
    //
    // Vector of pointers because they can be undefined.
    //
    std::vector<std::shared_ptr<math::Spline<float>>> _yToXCenter; // convert y pixel value to trace position
    std::vector<std::shared_ptr<math::Spline<float>>> _yToWavelength; // convert a y pixel value to wavelength

    void _set_xToFiberId();
    std::size_t _nKnot;                         // number of knots for splines
    //
    // An array that gives the fiberId half way up the chip
    //
    FiberMap _xToFiberId;
    //
    // offset (in pixels) for each trace in x, and y and in focus (microns at the slit); indexed by fiberIdx
    //
    Array2D _slitOffsets;

    lsst::afw::image::VisitInfo _visitInfo;
    /*
     * Private helper functions
     */
    void _setSplines(std::size_t fIndex,
                     Array1D const& xc, bool setXCenters,
                     Array1D const& wl, bool setWavelengths);

    void _setSplines(Array2D const&,
                     Array2D const&);
};

}}}

#endif
