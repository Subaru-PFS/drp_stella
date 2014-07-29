#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include "lsst/base.h"
#include "lsst/afw/geom/Box.h"
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
    std::vector<float> const & getProfile(int y) const { // really == 0 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        static std::vector<float> FAKE;
        return FAKE;
    }
private:
    lsst::afw::geom::Box2I _bbox;
    std::vector<float> _xcenters;
};

/************************************************************************************************************/
/**
 * \brief Describe a set of fiber traces
 */
class FiberTraceSet {
public:
    FiberTraceSet(int nAperture=0) : _traces(nAperture) {}
    virtual ~FiberTraceSet() {}

    int getNAperture() const { return _traces.size(); }
    FiberTrace &getFiberTrace(int const i) { return *_traces.at(i); }
    FiberTrace const& getFiberTrace(int const i) const { return *_traces.at(i); }
    void setFiberTrace(int const i, PTR(FiberTrace) trace);
    
private:
    std::vector<PTR(FiberTrace)> _traces;
};

}}}
#endif
