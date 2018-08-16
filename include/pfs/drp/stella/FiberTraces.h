#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include <tuple>

#include "ndarray.h"

#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"

#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {

std::string const fiberMaskPlane = "FIBERTRACE";  ///< Mask plane we care about

/**
 * @brief Describe a single fiber trace
 */
template<typename ImageT,
         typename MaskT=lsst::afw::image::MaskPixel,
         typename VarianceT=lsst::afw::image::VariancePixel>
class FiberTrace {
  public:
    typedef lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef lsst::afw::image::Image<ImageT> Image;
    typedef lsst::afw::image::Mask<MaskT> Mask;
    typedef lsst::afw::image::Image<VarianceT> Variance;

    /** @brief Class Constructors and Destructor
     *
     * @param maskedImage : maskedImage to set _trace to
     * @param xCenters : position of center for each row
     * @param fiberTraceId : FiberTrace ID
     * */
    explicit FiberTrace(MaskedImageT const& trace,
                        std::size_t fiberTraceId=0);

    /**
     * @brief Copy constructor (deep if required)
     *
     * @param fiberTrace : FiberTrace to copy
     * @param deep : Deep copy if true, shallow copy if false
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
               bool deep=false);
    
    FiberTrace(FiberTrace const&) = default;
    FiberTrace(FiberTrace &&) = default;
    FiberTrace & operator=(FiberTrace const&) = default;
    FiberTrace & operator=(FiberTrace &&) = default;

    /**
     * @brief Destructor
     */
    virtual ~FiberTrace() {}

    //@{
    /**
     * @brief Return the 2D MaskedImage of this fiber trace
     */
    MaskedImageT & getTrace() { return _trace; }
    MaskedImageT const& getTrace() const { return _trace; }
    //@}

    /**
     * @brief Extract the spectrum of this fiber trace using the profile
     */
    std::shared_ptr<Spectrum> extractSpectrum(
        MaskedImageT const& image, ///< image containing the spectrum
        bool fitBackground=false, ///< should I fit the background level?
        float clipNSigma=0, ///< clip data points at this many sigma (if > 0)
        bool useProfile=true  ///< use profile to perform "optimal" extraction?
    );

    /**
     * @brief Return an image containing the reconstructed 2D spectrum of the FiberTrace
     *
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    std::shared_ptr<Image> constructImage(Spectrum const& spectrum) const;
    
    /**
     * @brief set the ID number of this trace (_fiberId) to this number
     * @param fiberId : ID to be assigned to this FiberTrace
     */
    void setFiberId(std::size_t fiberId) { _fiberId = fiberId; }

    /**
     * @brief Return ID of this FiberTrace
     */
    std::size_t getFiberId() const { return _fiberId; }

  private:
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> _trace;
    std::size_t _fiberId;
};

/************************************************************************************************************/
/**
 * @brief Describe a set of fiber traces
 *
 */
template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
         typename VarianceT=lsst::afw::image::VariancePixel>
class FiberTraceSet {
  public:
    typedef lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef FiberTrace<ImageT, MaskT, VarianceT> FiberTraceT;
    typedef std::vector<std::shared_ptr<FiberTraceT>> Collection;
    typedef typename Collection::iterator iterator;
    typedef typename Collection::const_iterator const_iterator;

    explicit FiberTraceSet(
        std::size_t reservation,
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    ) : _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>()) {
        _traces.reserve(reservation);
    }

    explicit FiberTraceSet(
        Collection const& traces,
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    ) : _traces(traces),
        _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
    {}

    /**
     * @brief Copy constructor
     *
     * @param fiberTraceSet : If fiberTraceSet is not empty and deep is false,
     * the object shares ownership of fiberTraceSet's fiber trace vector and
     * increases the use count. If deep is true then a deep copy of each FiberTrace
     * is created.
     * If fiberTraceSet is empty, an empty object is constructed (as if default-constructed).
     * @param deep : See description of fiberTraceSet
     */
    explicit FiberTraceSet(FiberTraceSet const& fiberTraceSet, bool deep=false);

    FiberTraceSet(FiberTraceSet &&) = default;
    FiberTraceSet & operator=(FiberTraceSet const&) = default;
    FiberTraceSet & operator=(FiberTraceSet &&) = default;

    virtual ~FiberTraceSet() {}

    /**
     * @brief Return the number of apertures
     */
    std::size_t size() const { return _traces.size(); }

    //@{
    /// Get i-th trace
    ///
    /// No bounds checking.
    std::shared_ptr<FiberTraceT> operator[](const std::size_t i) { return _traces[i]; }
    std::shared_ptr<FiberTraceT> const operator[](const std::size_t i) const { return _traces[i]; }
    //@}

    //@{
    /// Get i-th trace
    ///
    /// Includes bounds checking.
    std::shared_ptr<FiberTraceT> get(const std::size_t i) { return _traces.at(i); }
    std::shared_ptr<FiberTraceT> const get(const std::size_t i) const { return _traces.at(i); }
    //@}

    /**
     * @brief Set the ith FiberTrace
     *
     * @param i : position in _traces which is to be replaced bye trace
     * @param trace : FiberTrace to replace _traces[i]
     */
    void set(std::size_t i, std::shared_ptr<FiberTraceT> trace) { _traces.at(i) = trace; }

    /**
     * @brief Add one FiberTrace to the set
     *
     * @param trace : FiberTrace to be added to _traces
     */
    void add(std::shared_ptr<FiberTraceT> trace) { _traces.push_back(trace); }

    /// Construct and add a FiberTrace
    template <class... Args>
    void add(Args&&... args) { _traces.push_back(std::make_shared<FiberTraceT>(args...)); }

    //@{
    /// Iterators
    iterator begin() { return _traces.begin(); }
    const_iterator begin() const { return _traces.begin(); }
    iterator end() { return _traces.end(); }
    const_iterator end() const { return _traces.end(); }
    //@}

    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() { return _metadata; }
    std::shared_ptr<lsst::daf::base::PropertySet const> getMetadata() const { return _metadata; }

    /**
     * @brief Return this->_traces
     */
    Collection const& getInternal() const { return _traces; }

    /**
     * @brief re-order the traces in _traces by the xCenter of each trace
     */
    void sortTracesByXCenter();

  private:
    Collection _traces; // traces for each aperture
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;  // FITS header
};

}}}

#endif
