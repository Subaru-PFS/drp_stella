///TODO: Add deep option to copy constructors
#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include <tuple>

#include "ndarray.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"

#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {
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
     * @param maskedImage : maskedImage to set _trace to
     * @param fiberTraceId : FiberTrace ID
     * */
    explicit FiberTrace(MaskedImageT const& maskedImage,
                        std::size_t fiberTraceId=0);

    /**
     * @brief Create a FiberTrace from a MaskedImage and a FiberTraceFunction
     * @param maskedImage : Masked CCD Image from which to extract the FiberTrace
     * @param fiberTraceFunction : FiberTraceFunction defining the FiberTrace
     * @param fiberId : set this number to this._fiberId
     */
    explicit FiberTrace(MaskedImageT const& maskedImage,
                        FiberTraceFunction const& fiberTraceFunction,
                        FiberTraceProfileFittingControl const& fiberTraceProfileFittingControl,
                        std::size_t fiberId=0);

    /**
     * @brief Copy constructor (deep if required)
     * @param fiberTrace : FiberTrace to copy
     * @param deep : Deep copy if true, shallow copy if false
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
               bool deep=false);
    
    /**
     * @brief Destructor
     */
    virtual ~FiberTrace() {}

    /**
     * @brief Return shared pointer to the 2D MaskedImage of this fiber trace
     */
    MaskedImageT & getTrace() { return _trace; }
    MaskedImageT const& getTrace() const { return _trace; }

    /**
     * @brief Extract the spectrum of this fiber trace using the _profile
     */
    std::shared_ptr<Spectrum> extractSpectrum(
        MaskedImageT const& image, ///< image containing the spectrum
        bool fitBackground=false, ///< should I fit the background level?
        float clipNSigma=0, ///< clip data points at this many sigma (if > 0)
        bool useProfile=true  ///< use profile to perform "optimal" extraction?
    );
    
    /**
     * @brief Return the fitted x-centers of the fiber trace
     */
    ndarray::Array<float const, 1, 1> getXCenters() const { return _xCenters; }

    /**
     * @brief Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    PTR(Image) constructImage(Spectrum const& spectrum) const;
    
    /**
     * @brief set the ID number of this trace (_fiberId) to this number
     * @param fiberId : ID to be assigned to this FiberTrace
     */
    void setFiberId(std::size_t fiberId) { _fiberId = fiberId; }

    /**
     * @brief Return ID of this FiberTrace
     */
    std::size_t getFiberId() const { return _fiberId; }

    FiberTraceFunction const& getFunction() const { return _function; }
    FiberTraceProfileFittingControl const& getFitting() const { return _fitting; }

    std::string const maskPlane = "FIBERTRACE";  ///< Mask plane we care about
  private:

    /**
     * @brief Calculate the spatial profile for the FiberTrace
     * Normally this would be a Flat FiberTrace, but in principle, if the spectrum
     * shows some kind of continuum, the spatial profile can still be calculated
     */
    void _calcProfile();

    /**
     * @brief Helper function for calcProfile, calculates profile for a swath
     * A swath is approximately FiberTraceProfileFittingControl.swathWidth long
     * Each swath is overlapping the previous swath for half of the swath width
     * spectrum:
     * |-----------------------------------------------------------------
     * swaths:
     * |---------------|--------------|--------------|--------------|----
     *         |---------------|--------------|--------------|-----------
     * @param swath : CCD image of the FiberTrace swath
     * @param xCentersSwath : 1D array containing the x center positions for the swath
     * @param iSwath : number of swath
     */
    ndarray::Array<float, 2, 1> _calcProfileSwath(
        lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& swath,
        ndarray::Array<float const, 1, 1> const& xCentersSwath,
        std::size_t iSwath
    );

    /**
     * @brief mark FiberTrace pixels in Mask image
     * @param value : value to Or into the FiberTrace mask
     */
    void _markFiberTraceInMask(MaskT value=1);

    /**
     * @brief Create _trace from maskedImage and _fiberTraceFunction
     * @param maskedImage : MaskedImage from which to extract the FiberTrace from
     * Pre: _xCenters set/calculated
     */
    void _createTrace(MaskedImageT const& maskedImage);

    /**
     * @brief Calculate boundaries for the swaths used for profile calculation
     * @param swathWidth_In : Approximate width for the swaths, will be adjusted
     * to fill the length of the FiberTrace with equally sized swaths
     * @return 2D array containing the pixel numbers for the start and the end
     * of each swath
     */
    ndarray::Array<std::size_t, 2, 1> _calcSwathBoundY(std::size_t swathWidth) const;

    /**
     * @brief : return _minCenMax (after recomputing if necessary)
     */
    ndarray::Array<std::size_t, 2, -2> _getMinCenMax();
    /**
     * @brief : Reconstruct _minCenMax from mask
     */
    void _reconstructMinCenMax();

    std::vector<std::shared_ptr<std::vector<float>>> _overSampledProfileFitXPerSwath;
    std::vector<std::shared_ptr<std::vector<float>>> _overSampledProfileFitYPerSwath;
    std::vector<std::shared_ptr<std::vector<float>>> _profileFittingInputXPerSwath;
    std::vector<std::shared_ptr<std::vector<float>>> _profileFittingInputYPerSwath;
    std::vector<std::shared_ptr<std::vector<float>>> _profileFittingInputXMeanPerSwath;
    std::vector<std::shared_ptr<std::vector<float>>> _profileFittingInputYMeanPerSwath;
    
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> _trace;
    ndarray::Array<float, 1, 1> _xCenters;
    ndarray::Array<std::size_t, 2, -2> _minCenMax;
    std::size_t _fiberId;
    FiberTraceFunction _function;
    FiberTraceProfileFittingControl _fitting;
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

    explicit FiberTraceSet(std::size_t reservation) { _traces.reserve(reservation); }

    /**
     * @brief Copy constructor
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
     * @param i : position in _traces which is to be replaced bye trace
     * @param trace : FiberTrace to replace _traces[i]
     */
    void set(
        std::size_t i,     ///< which aperture?
        std::shared_ptr<FiberTraceT> trace ///< the FiberTrace for the ith aperture
    ) {
        _traces.at(i) = trace;
    }

    /**
     * @brief Add one FiberTrace to the set
     * @param trace : FiberTrace to be added to _traces
     */
    void add(std::shared_ptr<FiberTraceT> trace) { _traces.push_back(trace); }
    template <class... Args>
    void add(Args&&... args) { _traces.push_back(std::make_shared<FiberTraceT>(args...)); }

    iterator begin() { return _traces.begin(); }
    const_iterator begin() const { return _traces.begin(); }
    iterator end() { return _traces.end(); }
    const_iterator end() const { return _traces.end(); }

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
};

}}}
#endif
