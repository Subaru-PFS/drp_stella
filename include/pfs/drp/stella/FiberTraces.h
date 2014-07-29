#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include "lsst/base.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/image/Image.h"
#include "lsst/pex/config.h"

namespace pfs { namespace drp { namespace stella {
/**
 * Control Fiber trace extraction
 */
struct FiberTraceControl {
    /// enum corresponding to legal values of interpolation string
    enum {  CHEBYSHEV, LEGENDRE, CUBIC, LINEAR, POLYNOMIAL } INTERPOLATION;
    
    LSST_CONTROL_FIELD(interpolation, std::string, "Interpolation schemes");
    LSST_CONTROL_FIELD(nApertures, int, "");
    LSST_CONTROL_FIELD(overSample, int, "");
    LSST_CONTROL_FIELD(maxIterSF, int, "");
    LSST_CONTROL_FIELD(maxIterSky, int, "");
    LSST_CONTROL_FIELD(maxIterSig, int, "");
    LSST_CONTROL_FIELD(apertureFWHM, double, "");
    LSST_CONTROL_FIELD(signalThreshold, double, "");   // Should we use lsst::afw::detection::Threshold?
    LSST_CONTROL_FIELD(maxNumberOfAperturesToBeFound, int, "");

    FiberTraceControl() :
        interpolation("CHEBYSHEV"),
        nApertures(0),
        overSample(0),
        maxIterSF(0),
        maxIterSky(0),
        maxIterSig(0),
        apertureFWHM(0),
        signalThreshold(0),
        maxNumberOfAperturesToBeFound(0) {}
};

/**
 * \brief Describe a single fiber trace
 */
class FiberTrace {
public:
    FiberTrace(lsst::afw::geom::Box2I const& bbox);
    virtual ~FiberTrace() {}

    /// Return the bounding box of this fiber trace
    lsst::afw::geom::Box2I getBBox() const { return _bbox; }
    /// Set the bounding box of this fiber trace to bbox
    void setBBox(lsst::afw::geom::Box2I const& bbox) ///< Desired bounding box
        { _bbox = bbox; }

    /// Return the x-centers of the fiber trace
    ///
    /// N.b. These are relative to getBBox().getBeginX()
    std::vector<float> getXCenters() const { return _xcenters; }
    /// Set the x-center of the fiber trace
    void setXCenters(std::vector<float> xcenters) ///< New value of x-center
        { _xcenters = xcenters; }

    /// Return the profile for a row
    /// N.b. the first value corresponds to the minimum x-value of the trace (see getBBox)
    virtual ndarray::Array<float const, 1, 1> getProfile(int y) const = 0;
    /// Return the profile for a row
    /// N.b. the first value corresponds to the minimum x-value of the trace (see getBBox)
    virtual ndarray::Array<float, 1, 1> getProfile(int y) = 0;
private:
    lsst::afw::geom::Box2I _bbox;
    std::vector<float> _xcenters;
};

/************************************************************************************************************/
class FiberTraceSet;

/**
 * \brief A FiberTrace that uses an image to save the line-by-line profiles
 */
class ImageFiberTrace : public FiberTrace {
public:
    ImageFiberTrace(lsst::afw::geom::Box2I const& bbox);

    /// Return the profile for a row
    /// N.b. the first value corresponds to the minimum x-value of the trace (see getBBox)
    virtual ndarray::Array<float, 1, 1> getProfile(int y);
    /// Return the profile for a row
    /// N.b. the first value corresponds to the minimum x-value of the trace (see getBBox)
    virtual ndarray::Array<float const, 1, 1> getProfile(int y) const;
private:
    friend class FiberTraceSet;
    PTR(lsst::afw::image::Image<float>) _profArray;
};

/************************************************************************************************************/
/**
 * \brief Describe a set of fiber traces
 *
 * \note If a profArray is passed to the ctor, you are required to pass ImageFiberTraces to setFiberTrace
 */
class FiberTraceSet {
public:
    /// Create a FiberTraceSet that uses an image to save the profiles
    explicit FiberTraceSet(PTR(lsst::afw::image::Image<float>) profArray) ///< profile array
        : _traces(0), _profArray(profArray) {}

    explicit FiberTraceSet(int nAperture=0, ///< number of apertures
                           PTR(lsst::afw::image::Image<float>)
                           profArray=PTR(lsst::afw::image::Image<float>)()) ///< profile array
        : _traces(nAperture), _profArray(profArray) {}

    virtual ~FiberTraceSet() {}

    /// Return the number of apertures
    int getNAperture() const { return _traces.size(); }
    /// Return the FiberTrace for the ith aperture
    FiberTrace &getFiberTrace(int const i ///< desired aperture
                             ) { return *_traces.at(i); }
    FiberTrace const& getFiberTrace(int const i ///< desired aperture
                                   ) const { return *_traces.at(i); }
    /// Set the ith FiberTrace
    void setFiberTrace(int const i,     ///< which aperture?
                       PTR(FiberTrace) trace ///< the FiberTrace for the ith aperture
                      );
private:
    std::vector<PTR(FiberTrace)> _traces; // traces for each aperture
    PTR(lsst::afw::image::Image<float>) _profArray; // Image defining apertures iff provided
};

}}}
#endif
