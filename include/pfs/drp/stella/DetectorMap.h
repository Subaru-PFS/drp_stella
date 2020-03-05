#if !defined(PFS_DRP_STELLA_DETECTORMAP_H)
#define PFS_DRP_STELLA_DETECTORMAP_H

#include <vector>
#include "ndarray.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/image/VisitInfo.h"
#include "lsst/afw/table/io/Persistable.h"

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
class DetectorMap : public lsst::afw::table::io::Persistable {
  public:
    enum ArrayRow { DX = 0, DY = 1, DFOCUS = 2 };

    using FiberMap = ndarray::Array<int const, 1, 1>;
    using Array2D = ndarray::Array<float, 2, 1>;
    using Array1D = ndarray::Array<float, 1, 1>;
    using VisitInfo = lsst::afw::image::VisitInfo;
    using Spline = math::Spline<float>;

    DetectorMap(lsst::geom::Box2I bbox,  // detector's bounding box
                FiberMap const& fiberId,  // 1-indexed IDs for each fibre
                std::vector<ndarray::Array<float, 1, 1>> const& centerKnots,
                std::vector<ndarray::Array<float, 1, 1>> const& centerValues,
                std::vector<ndarray::Array<float, 1, 1>> const& wavelengthKnots,
                std::vector<ndarray::Array<float, 1, 1>> const& wavelengthValues,
                Array2D const& slitOffsets=Array2D(),  // per-fibre offsets
                VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),  // Visit information
                std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr  // FITS header
                );

    DetectorMap(lsst::geom::Box2I bbox,  // detector's bounding box
                FiberMap const& fiberId,  // 1-indexed IDs for each fibre
                std::vector<std::shared_ptr<DetectorMap::Spline const>> const& center,
                std::vector<std::shared_ptr<DetectorMap::Spline const>> const& wavelength,
                Array2D const& slitOffsets=Array2D(),  // per-fibre offsets
                VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),  // Visit information
                std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr  // FITS header
                );

    virtual ~DetectorMap() {}
    DetectorMap(DetectorMap const&) = default;
    DetectorMap(DetectorMap &&) = default;
    DetectorMap & operator=(DetectorMap const&) = default;
    DetectorMap & operator=(DetectorMap &&) = default;

    /** \brief return the bbox */
    lsst::geom::Box2I getBBox() const { return _bbox; }

    /** \brief return the fiberIds */
    FiberMap & getFiberId() { return _fiberId; }
    FiberMap const& getFiberId() const { return _fiberId; }

    /** \brief Return the number of fibers */
    std::size_t getNumFibers() const { return _fiberId.size(); }

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
    std::vector<Array1D> getWavelength() const;
    float getWavelength(std::size_t fiberId, ///< fiberId
                        float y              ///< desired y value
                        ) const;

    /**
     * Set the wavelength values for a fibre
     */
    void setWavelength(std::size_t fiberId,  ///< Fiber identifier
                       Array1D const& wavelength  //< wavelength for each row
                       );
    void setWavelength(std::size_t fiberId,                       ///< 1-indexed fiberID 
                       Array1D const& knots,  ///< knots for wavelength
                       Array1D const& wavelength ///< wavelengths for fibre
                      );

    //@{
    /**
     * Return the xCenter values for a fibre
     */
    Array1D getXCenter(std::size_t fiberId ///< fiberId
                                          ) const;
    std::vector<Array1D> getXCenter() const;
    float getXCenter(std::size_t fiberId, ///< fiberId
                     float y              ///< desired y value
                     ) const;
    //@}

    /**
     * Set the xCenter values for a fibre
     */
    void setXCenter(std::size_t fiberId,  ///< fiber identifier
                    Array1D const& xCenter  ///< center for each row
                    );
    void setXCenter(std::size_t fiberId,                       ///< 1-indexed fiberID 
                    Array1D const& knots,  ///< knots for center
                    Array1D const& xCenter ///< center of trace for fibre
                   );

    /** \brief
     * Return the fiberId given a position on the detector
     */
    int findFiberId(lsst::geom::PointD pixelPos ///< position on detector
                   ) const;

    /** \brief
     * Return the position of the fiber trace on the detector, given a fiberId and wavelength
     */
    lsst::geom::PointD findPoint(int fiberId,               ///< Desired fibreId
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

    VisitInfo getVisitInfo() const { return _visitInfo; }
    void setVisitInfo(VisitInfo &visitInfo) { _visitInfo = visitInfo; };

    math::Spline<float> const& getCenterSpline(std::size_t index) const;
    math::Spline<float> const& getWavelengthSpline(std::size_t index) const;

    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() { return _metadata; }
    std::shared_ptr<lsst::daf::base::PropertySet const> getMetadata() const { return _metadata; }

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const { return "DetectorMap"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:                              // initialise before _yTo{XCenter,Wavelength}
    std::size_t _nFiber;                // number of fibers
    lsst::geom::Box2I _bbox;       // bounding box of detector
    FiberMap _fiberId;         // The fiberIds (between 1 and c. 2400) present on this detector

    //
    // These std::vectors are indexed by fiberIndex (not fiberId)
    //
    // Vector of pointers because they can be undefined.
    //
    std::vector<std::shared_ptr<Spline const>> _yToXCenter; // convert y pixel value to trace position
    std::vector<std::shared_ptr<Spline const>> _yToWavelength; // convert a y pixel value to wavelength

    void _set_xToFiberId();
    //
    // An array that gives the fiberId half way up the chip
    //
    ndarray::Array<int, 1, 1> _xToFiberId;
    //
    // offset (in pixels) for each trace in x, and y and in focus (microns at the slit); indexed by fiberIdx
    //
    Array2D _slitOffsets;

    lsst::afw::image::VisitInfo _visitInfo;
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;  // FITS header
};

}}}

#endif
