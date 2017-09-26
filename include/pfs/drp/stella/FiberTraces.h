///TODO: Add deep option to copy constructors
#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>

#include "ndarray.h"

#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Spectra.h"

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
    typedef Spectrum<ImageT, MaskT, VarianceT> SpectrumT;


    /** @brief Class Constructors and Destructor
     * @param maskedImage : maskedImage to set _trace to
     * @param fiberTraceId : FiberTrace ID
     * */
    explicit FiberTrace(PTR(const MaskedImageT) const& maskedImage,
                        std::size_t fiberTraceId = 0);

    /**
     * @brief Create a FiberTrace from a MaskedImage and a FiberTraceFunction
     * @param maskedImage : Masked CCD Image from which to extract the FiberTrace
     * @param fiberTraceFunction : FiberTraceFunction defining the FiberTrace
     * @param iTrace : set this number to this._iTrace
     */
    explicit FiberTrace(PTR(const MaskedImageT) const& maskedImage,
                        PTR(const FiberTraceFunction) const& fiberTraceFunction, 
                        PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl,
                        std::size_t iTrace=0);

    /**
     * @brief Copy constructor (deep if required)
     * @param fiberTrace : FiberTrace to copy
     * @param deep : Deep copy if true, shallow copy if false
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace, bool const deep=false);
    
    /**
     * @brief Destructor
     */
    virtual ~FiberTrace() {}

    /**
     * @brief Return shared pointer to the 2D MaskedImage of this fiber trace
     * not const
     */
    PTR(MaskedImageT) getTrace() { return _trace; }

    /**
     * @brief Return shared pointer to the 2D MaskedImage of this fiber trace
     * const
     */
    const PTR(const MaskedImageT) getTrace() const { return _trace; }

    /**
     * @brief Extract the spectrum of this fiber trace using the _profile
     */
    PTR(SpectrumT) extractFromProfile(PTR(const MaskedImageT) const& spectrumImage);
    
    /**
     * @brief Simple Sum Extraction of this fiber trace
     */
    PTR(SpectrumT) extractSum(PTR(const MaskedImageT) const& spectrumImage);
    
    /**
     * @brief Return the fitted x-centers of the fiber trace
     */
    const ndarray::Array< float, 1, 1 > getXCenters() const { return _xCenters; }

    /**
     * @brief Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    PTR(Image) getReconstructed2DSpectrum(SpectrumT const& spectrum) const;
    
    /**
     * @brief set the ID number of this trace (_iTrace) to this number
     * @param iTrace : ID to be assigned to this FiberTrace
     */
    void setITrace(const std::size_t iTrace){_iTrace = iTrace;}

    /**
     * @brief Return ID of this FiberTrace
     */
    std::size_t getITrace() const {return _iTrace;}

    /**
     * @brief: compare x and y center of fiberTrace to xCenters and yCenters to identify and set the _traceID
     * @param xCenters: x centers per fiber per row, shape(nfibers * nRows)
     * @param fiberIds: fiber ID, shape(nfibers * nRows)
     * @param nTraces: number of fiber traces on CCD
     * @param nRows: number of CCD rows
     * @param startPos: fiber number to start searching
     * @return void
     */
    void assignTraceID(ndarray::Array<float, 1, 1> const& xCenters,
                       ndarray::Array<int, 1, 1> const& fiberIds,
                       std::size_t nTraces,
                       std::size_t nRows);

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
     * @param imageSwath : array containing the CCD image of the FiberTrace swath
     * @param maskSwath : array containing the mask of the FiberTrace swath
     * @param varianceSwath : array containing the variance of the FiberTrace swath
     * @param xCentersSwath : 1D array containing the x center positions for the swath
     * @param iSwath : number of swath
     */
    ndarray::Array<float, 2, 1> _calcProfileSwath(ndarray::Array<ImageT const, 2, 1> const& imageSwath,
                                                 ndarray::Array<MaskT const, 2, 1> const& maskSwath,
                                                 ndarray::Array<VarianceT const, 2, 1> const& varianceSwath,
                                                 ndarray::Array<float const, 1, 1> const& xCentersSwath,
                                                 std::size_t const iSwath);

    /**
     * @brief mark FiberTrace pixels in Mask image
     * @param value : value to Or into the FiberTrace mask
     */
    void _markFiberTraceInMask(MaskT value = 1);

    /**
     * @brief Create _trace from maskedImage and _fiberTraceFunction
     * @param maskedImage : MaskedImage from which to extract the FiberTrace from
     * Pre: _xCenters set/calculated
     */
    void _createTrace(PTR(const MaskedImageT) const& maskedImage);

    /**
     * @brief Calculate boundaries for the swaths used for profile calculation
     * @param swathWidth_In : Approximate width for the swaths, will be adjusted
     * to fill the length of the FiberTrace with equally sized swaths
     * @return 2D array containing the pixel numbers for the start and the end
     * of each swath
     */
    ndarray::Array<std::size_t, 2, 1> _calcSwathBoundY(std::size_t const& swathWidth_In) const;

    /**
     * @brief : return _minCenMax (after recomputing if necessary)
     */
    ndarray::Array<size_t, 2, -2> _getMinCenMax();
    /**
     * @brief : Reconstruct _minCenMax from mask
     */
    void _reconstructMinCenMax();

    std::vector<PTR(std::vector<float>)> _overSampledProfileFitXPerSwath;
    std::vector<PTR(std::vector<float>)> _overSampledProfileFitYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXMeanPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYMeanPerSwath;
    
    PTR(lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT>) _trace;
    ndarray::Array<float, 1, 1> _xCenters;
    ndarray::Array<size_t, 2, -2> _minCenMax;
    std::size_t _iTrace;
    PTR(const FiberTraceFunction) _fiberTraceFunction;
    PTR(FiberTraceProfileFittingControl) _fiberTraceProfileFittingControl;

  protected:
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
    typedef Spectrum<ImageT, MaskT, VarianceT> SpectrumT;
    typedef std::vector<PTR(FiberTraceT)> Collection;

    /**
     * Class Constructors and Destructor
     */
    
    /**
     * @brief Creates a new FiberTraceSet object of size nTraces
     * @param nTraces : Size (length) of the new FiberTraceSet
     */
    explicit FiberTraceSet()
            : _traces(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(0)){};

    /**
     * @brief Copy constructor
     * @param fiberTraceSet : If fiberTraceSet is not empty and deep is false,
     * the object shares ownership of fiberTraceSet's fiber trace vector and
     * increases the use count. If deep is true then a deep copy of each FiberTrace
     * is created.
     * If fiberTraceSet is empty, an empty object is constructed (as if default-constructed).
     * @param deep : See description of fiberTraceSet
     */
    explicit FiberTraceSet(const FiberTraceSet &fiberTraceSet, bool const deep = false);
        
    /**
     * @brief Destructor
     */
    virtual ~FiberTraceSet() {}

    /**
     * @brief Return the number of apertures
     */
    std::size_t getNtrace() const { return _traces->size(); }

    /**
     * @brief Return the FiberTrace for the ith aperture
     * @param i : Position in _traces from which to return the FiberTrace
     */
    PTR(FiberTraceT) getFiberTrace(const std::size_t i);

    /**
     * @brief Return a copy of the FiberTrace for the ith aperture
     * @param i : Position in _traces from which to return a copy of the FiberTrace
     */
    PTR(FiberTraceT) const getFiberTrace(const std::size_t i) const;

    /**
     * @brief Set the ith FiberTrace
     * @param i : position in _traces which is to be replaced bye trace
     * @param trace : FiberTrace to replace _traces[i]
     */
    void setFiberTrace(const std::size_t i,     ///< which aperture?
                       const PTR(FiberTraceT) &trace ///< the FiberTrace for the ith aperture
                      );

    /**
     * @brief Add one FiberTrace to the set
     * @param trace : FiberTrace to be added to _traces
     * @param iTrace : if != 0, set the ID of trace to this number
     */
    void addFiberTrace(const PTR(FiberTraceT) &trace, const std::size_t iTrace = 0);

    /**
     * @brief Return this->_traces
     */
    PTR(Collection) getTraces() const { return _traces; }

    /**
     * @brief re-order the traces in _traces by the xCenter of each trace
     */
    void sortTracesByXCenter();

    /**
     * @brief: assign trace number to set of FiberTraces from x and y center by comparing
     *         the center position to the center positions of the zemax model
     * @param traceIds: shape(nfibers * nRows) from zemax model
     * @param xCenters: shape(nfibers * nRows) from zemax model
     */
    void assignTraceIDs(ndarray::Array<int, 1, 1> const& fiberIds,
                        ndarray::Array<float, 1, 1> const& xCenters);

  private:
    PTR(Collection) _traces; // traces for each aperture
};

namespace math{
  /** 
   * @brief identifies and traces the fiberTraces in maskedImage, and extracts them into individual FiberTraces
   * FiberTraces in returned FiberTraceSet will be sorted by their xCenter positions
   * Set I_NTermsGaussFit to
   *       1 to look for maximum only without GaussFit
   *       3 to fit Gaussian
   *       4 to fit Gaussian plus constant (sky)
   *         Spatial profile must be at least 5 pixels wide
   *       5 to fit Gaussian plus linear term (sloped sky)
   *         Spatial profile must be at least 6 pixels wide
   * NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
   * @param maskedImage : MaskedImage in which to find and trace the FiberTraces
   * @param fiberTraceFunctionFindingControl : Control to be used in task
   **/
  template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
           typename VarianceT=lsst::afw::image::VariancePixel>
  PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(
                    PTR(const lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT>) const& maskedImage,
                    PTR(const FiberTraceFunctionFindingControl) const& fiberTraceFunctionFindingControl,
                    PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);
    
  struct FindCenterPositionsOneTraceResult{
      std::vector<float> apertureCenterIndex;
      std::vector<float> apertureCenterPos;
      std::vector<float> eApertureCenterPos;
  };
  
  /**
   * @brief: traces the fiberTrace closest to the bottom of the image and sets it to zero
   * @param ccdImage: image to trace
   * @param ccdImageVariance: variance of image to trace (used for fitting)
   * @param fiberTraceFunctionFindingControl: parameters to find and trace a fiberTrace
   * */
  template<typename ImageT, typename VarianceT=lsst::afw::image::VariancePixel>
  FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
	PTR(lsst::afw::image::Image<ImageT>) & ccdImage,
        PTR(lsst::afw::image::Image<VarianceT>) & ccdImageVariance,
        PTR(const FiberTraceFunctionFindingControl) const& fiberTraceFunctionFindingControl);
  
  /**
   * @brief: returns ndarray containing the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
   *         NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
   * @param fiberTraceFunctionIn : FiberTraceFunction to use when calculating the xCenters
   * @param ccdHeightIn : Not used, remove
   * @param ccdWidthIn : Not used, remove
   */
  ndarray::Array<float, 1, 1> calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                 std::size_t const& ccdHeightIn = 0,
                                                 std::size_t const& ccdWidthIn = 0);

  /**
   * @brief: returns ndarray containing the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
   *         NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
   * @param fiberTraceFunctionIn : FiberTraceFunction to use when calculating the xCenters
   * @param yIn : This range in y will be converted to [-1.0,1.0] when calculating the xCenters
   * @param ccdHeightIn : Not used, remove
   * @param ccdWidthIn : Not used, remove
   */
  ndarray::Array<float, 1, 1> calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                 ndarray::Array<float, 1, 1> const& yIn,
                                                 std::size_t const& ccdHeightIn = 0,
                                                 std::size_t const& ccdWidthIn = 0);
}

}}}
#endif
