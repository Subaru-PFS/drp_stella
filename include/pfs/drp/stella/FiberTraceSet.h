#if !defined(PFS_DRP_STELLA_FIBERTRACESET_H)
#define PFS_DRP_STELLA_FIBERTRACESET_H

#include <memory>
#include <vector>

#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/FiberTrace.h"
#include "pfs/drp/stella/SpectrumSet.h"

namespace pfs { namespace drp { namespace stella {

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
    std::shared_ptr<FiberTraceT> operator[](std::ptrdiff_t i) { return _traces[i]; }
    std::shared_ptr<FiberTraceT> const operator[](std::ptrdiff_t i) const { return _traces[i]; }
    //@}

    //@{
    /// Get i-th trace
    ///
    /// Includes bounds checking.
    std::shared_ptr<FiberTraceT> get(std::ptrdiff_t i) { return _traces.at(i); }
    std::shared_ptr<FiberTraceT> const get(std::ptrdiff_t i) const { return _traces.at(i); }
    //@}

    /**
     * @brief Set the ith FiberTrace
     *
     * @param i : position in _traces which is to be replaced bye trace
     * @param trace : FiberTrace to replace _traces[i]
     */
    void set(std::ptrdiff_t i, std::shared_ptr<FiberTraceT> trace) { _traces.at(i) = trace; }

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

    /**
     * @brief Extract spectra from an image
     *
     * We perform a simultaneous optimal extraction for all the traces using
     * least-squares on each row.
     *
     * @param image : Image containing the spectra
     * @param badBitMask : ignore pixels where (value & badBitMask) != 0
     * @param minFracMask : minimum fractional contribution of pixel for mask to be accumulated
     * @return extracted spectra.
     */
    SpectrumSet extractSpectra(MaskedImageT const& image, MaskT badBitMask=0, float minFracMask=0.3) const;

  private:
    Collection _traces; // traces for each aperture
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;  // FITS header
};

}}}

#endif
