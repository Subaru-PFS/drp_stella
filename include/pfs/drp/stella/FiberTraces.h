///TODO: Add deep option to copy constructors
#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <iostream>
#include <vector>

#include "Controls.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Exception.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "math/Chebyshev.h"
#include "math/CurveFitting.h"
#include "math/Math.h"
#include "Spectra.h"
#include "spline.h"
#include "utils/Utils.h"

namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;

using namespace std;

namespace pfs { namespace drp { namespace stella {
/**
 * @brief Describe a single fiber trace
 */
template<typename ImageT,
         typename MaskT=afwImage::MaskPixel,
         typename VarianceT=afwImage::VariancePixel>
class FiberTrace {
  public:
    typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef afwImage::Image<ImageT> Image;
    typedef afwImage::Mask<MaskT> Mask;
    typedef afwImage::Image<VarianceT> Variance;
    typedef Spectrum<ImageT, MaskT, VarianceT, VarianceT> SpectrumT;


    /** @brief Class Constructors and Destructor
     * @param width : width of FiberTrace (number of columns)
     * @param height : height of FiberTrace (number of rows)
     * @param iTrace : FiberTrace number
     * */
    explicit FiberTrace(std::size_t width = 0, std::size_t height = 0, std::size_t iTrace = 0);

    /**
     * @brief Create a FiberTrace form a MaskedImage and a FiberTraceFunction
     * @param maskedImage : Masked CCD Image from which to extract the FiberTrace
     * @param fiberTraceFunction : FiberTraceFunction defining the FiberTrace
     * @param iTrace : set this number to this._iTrace
     */
    explicit FiberTrace(PTR(const MaskedImageT) const& maskedImage,
                        PTR(const FiberTraceFunction) const& fiberTraceFunction, 
                        std::size_t iTrace=0);
    
    /**
     * @brief Copy constructor (shallow)
     * @param fiberTrace : FiberTrace to copy (shallow)
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace);

    /**
     * @brief Copy constructor (deep if required)
     * @param fiberTrace : FiberTrace to copy
     * @param deep : Deep copy if true, shallow copy if false
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace, bool const deep);
    
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
     * @brief Set the 2D image of this fiber trace to imageTrace
     * @param trace : FiberTrace to copy (shallow) to this
     * Pre: _fiberTraceFunction must be set
     */
    void setTrace(PTR(MaskedImageT) & trace);// { _trace = trace; }

    /**
     * @brief Return the image of the spatial profile
     */
    PTR(afwImage::Image<float>) getProfile() const{ return _profile; }

    /**
     * @brief Set the _profile of this fiber trace to profile
     * @param profile : Profile to set _profile to
     */
    void setProfile( PTR(afwImage::Image<float>) const& profile);

    /**
     * @brief Extract the spectrum of this fiber trace using the _profile
     */
    PTR(SpectrumT) extractFromProfile();
    
    /**
     * @brief Simple Sum Extraction of this fiber trace
     */
    PTR(SpectrumT) extractSum();

    /**
     * @brief Create _trace from maskedImage and _fiberTraceFunction
     * @param maskedImage : MaskedImage from which to extract the FiberTrace from
     * Pre: _xCenters set/calculated
     */
    void createTrace(PTR(const MaskedImageT) const& maskedImage);

    /**
     * @brief Return _fiberTraceFunction (const)
     */
    const PTR(const FiberTraceFunction) getFiberTraceFunction() const { return _fiberTraceFunction; }

    /**
     * @brief set _fiberTraceFunction to fiberTraceFunction
     * @param fiberTraceFunction : FiberTraceFunction to copy to _fiberTraceFunction
     */
    void setFiberTraceFunction(PTR(FiberTraceFunction const) fiberTraceFunction);

    /**
     * @brief Return _fiberTraceProfileFittingControl
     */
    PTR(FiberTraceProfileFittingControl) getFiberTraceProfileFittingControl() const { return _fiberTraceProfileFittingControl; }

    /**
     * @brief copy (shallow) input to _fiberTraceProfileFittingControl
     * @param fiberTraceProfileFittingControl : use this FiberTraceProfileFittingControl
     */
    void setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);// { _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl; }

    /**
     * @brief copy (deep) input to _fiberTraceProfileFittingControl
     * @param fiberTraceProfileFittingControl : use this FiberTraceProfileFittingControl
     */
    void setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl const) const& fiberTraceProfileFittingControl);// { _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl; }
    
    /**
     * @brief Return the fitted x-centers of the fiber trace
     */
    const ndarray::Array< float, 1, 1 > getXCenters() const { return _xCenters; }

    /**
     * @brief Set _xCenters (fitted xCenter values) to input
     * @param xCenters : Copy (deep) to _xCenters
     */
    void setXCenters( ndarray::Array< float, 1, 1 > const& xCenters );

    /**
     * @brief Return the measured x-centers of the fiber trace
     */
    ndarray::Array< float, 2, 1 > getXCentersMeas() const { return _xCentersMeas; }

    /**
     * @brief Set _xCentersMeas (measured xCenter values) to input
     * @param xCentersMeas : Copy (deep) to _xCentersMeas
     */
    void setXCentersMeas( ndarray::Array< float, 2, 1 > const& xCentersMeas);

    /**
     * @brief Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    PTR(Image) getReconstructed2DSpectrum(SpectrumT const& spectrum) const;

    /**
     * @brief Return shared pointer to an image containing the reconstructed background of the FiberTrace
     * @param backgroundSpectrum : 1D spectrum to reconstruct the 2D image from
     */
    PTR(Image) getReconstructedBackground(SpectrumT const& backgroundSpectrum) const;

    /**
     * @brief Return shared pointer to an image containing the reconstructed 2D spectrum + background of the FiberTrace
     * @param spectrum : 1D spectrum to use for the 2D reconstruction
     * @param background : 1D background spectrum to use for the 2D reconstruction
     */
    PTR(Image) getReconstructed2DSpectrum(SpectrumT const& spectrum,
                                          SpectrumT const& background) const;
    
    /**
     * @brief Calculate the spatial profile for the FiberTrace
     * Normally this would be a Flat FiberTrace, but in principle, if the spectrum
     * shows some kind of continuum, the spatial profile can still be calculated
     */
    void calcProfile();

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
    ndarray::Array<float, 2, 1> calcProfileSwath(ndarray::Array<ImageT const, 2, 1> const& imageSwath,
                                                 ndarray::Array<MaskT const, 2, 1> const& maskSwath,
                                                 ndarray::Array<VarianceT const, 2, 1> const& varianceSwath,
                                                 ndarray::Array<float const, 1, 1> const& xCentersSwath,
                                                 std::size_t const iSwath);

    /**
     * @brief Calculate boundaries for the swaths used for profile calculation
     * @param swathWidth_In : Approximate width for the swaths, will be adjusted
     * to fill the length of the FiberTrace with equally sized swaths
     * @return 2D array containing the pixel numbers for the start and the end
     * of each swath
     */
    ndarray::Array<std::size_t, 2, 1> calcSwathBoundY(const std::size_t swathWidth_In) const;
    
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
     * @brief Check if _trace is set
     */
    bool isTraceSet() const {return _isTraceSet;}

    /**
     * @brief Check if the spatial profile (_profile) has been calculated
     */
    bool isProfileSet() const {return _isProfileSet;}

    /**
     * @brief Check if _fiberTraceProfileFittingControl has been set
     */
    bool isFiberTraceProfileFittingControlSet() const {return _isFiberTraceProfileFittingControlSet;}

    /**
     * @brief Return width of this FiberTrace
     */
    std::size_t getWidth() const {return _trace->getImage()->getWidth();}

    /**
     * @brief Return height of this FiberTrace
     */
    std::size_t getHeight() const {return _trace->getImage()->getHeight();}

    /**
     * @brief Return the coefficients of the trace function of the xCenters
     */
    ndarray::Array<float, 1, 1> getTraceCoefficients() const;

    /**
     * @brief Return a smart pointer to this FiberTrace
     */
    PTR(FiberTrace) getPointer();

  private:
    std::vector<PTR(std::vector<float>)> _overSampledProfileFitXPerSwath;
    std::vector<PTR(std::vector<float>)> _overSampledProfileFitYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXMeanPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYMeanPerSwath;
    
    ///TODO: replace variables with smart pointers?????
    PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) _trace;
    PTR(afwImage::Image<float>) _profile;
    ndarray::Array<float, 2, 1> _xCentersMeas;
    ndarray::Array<float, 1, 1> _xCenters;
    std::size_t _iTrace;
    bool _isTraceSet;
    bool _isProfileSet;
    bool _isFiberTraceProfileFittingControlSet;
    PTR(const FiberTraceFunction) _fiberTraceFunction;
    PTR(FiberTraceProfileFittingControl) _fiberTraceProfileFittingControl;

  protected:
};

/************************************************************************************************************/
/**
 * @brief Describe a set of fiber traces
 *
 */
template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
class FiberTraceSet {
  public:
    typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef FiberTrace<ImageT, MaskT, VarianceT> FiberTraceT;
    typedef Spectrum<ImageT, MaskT, VarianceT, VarianceT> SpectrumT;
    typedef std::vector<PTR(FiberTraceT)> Collection;

    /**
     * Class Constructors and Destructor
     */
    
    /**
     * @brief Creates a new FiberTraceSet object of size nTraces
     * @param nTraces : Size (length) of the new FiberTraceSet
     */
    explicit FiberTraceSet(std::size_t nTraces=0);

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
    std::size_t size() const { return _traces->size(); }
    
    /*
     * @brief Extract FiberTraces from new MaskedImage
     *        NOTE that this changes this FiberTraceSet!
     * @param maskedImage in: MaskedImage from which to extract the FiberTraces
    */
    void createTraces(const PTR(const MaskedImageT) &maskedImage);

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
     * @brief Removes from the vector either a single element (position) or a range of elements ([first,last)).
     * This effectively reduces the container size by the number of elements removed, which are destroyed.
     * @param iStart : FiberTrace number to remove if iEnd == 0, otherwise starting position of range to be removed
     * @param iEnd : if != 0 end + 1 of range of FiberTraces to be removed from _traces
     */
    void erase(const std::size_t iStart, const std::size_t iEnd=0);

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
     * @brief Set this->_fiberTraceProfileFittingControl
     * @param fiberTraceProfileFittingControl : FiberTraceProfileFittingControl to use for this
     */
    void setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);

    /**
     * @brief Set profiles of all traces in this FiberTraceSet to respective FiberTraces in input set
     * NOTE: the FiberTraces should be sorted by their xCenters before performing this operation!
     * @param fiberTraceSet : use profiles from that FiberTraceSet for this
     */
    void setAllProfiles(const PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) &fiberTraceSet);

    /**
     * @brief re-order the traces in _traces by the xCenter of each trace
     */
    void sortTracesByXCenter();

    ///TODO:
    /// Extract spectrum and background for one slit spectrum
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(Spectrum<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground(const std::size_t traceNumber)

    ///TODO:
    /// Extract spectrum and background for all slit spectra
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(SpectrumSet<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground()

    /**
     * @brief calculate profiles for all traces
     */
    void calcProfileAllTraces();
    
    /**
     * @brief 'optimally' extract 1D spectrum from previously provided profile
     * @param traceNumber : 'optimally' extract _traces[traceNumber] from its profile
     */
    PTR(SpectrumT) extractTraceNumberFromProfile(const std::size_t traceNumber);

    /**
     * @brief 'optimally' extract 1D spectra from their profiles
     */
    PTR(SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) extractAllTracesFromProfile();

    ///TODO:
    /// Extract spectrum and background for one slit spectrum
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(Spectrum<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackgroundFromProfile(int traceNumber)

    ///TODO:
    /// Extract spectrum and background for all slit spectra
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(SpectrumSet<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackgroundFromProfile()

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
  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
  PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                     const PTR(const FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl);
  
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
  template<typename ImageT, typename VarianceT=afwImage::VariancePixel>
  FindCenterPositionsOneTraceResult findCenterPositionsOneTrace( PTR(afwImage::Image<ImageT>) & ccdImage,
                                                                 PTR(afwImage::Image<VarianceT>) & ccdImageVariance,
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

  /**
   * @brief : extract a wide flatFiberTrace, fit profile, normalize, reduce width
   * @param maskedImage : CCD image dithered Flat 
   * @param fiberTraceFunctionWide : FiberTraceFunction (wide) for dithered flat
   * @param fiberTraceFunctionControlNarrow : FiberTraceFunctionControl (narrow) for output FiberTrace
   * @param fiberTraceProfileFittingControl : ProfileFittingControl for fitting the spatial profile of the dithered Flat
   * @param minSNR : normalized pixel values with an SNR lower than minSNR are set to 1.
   * @param iTrace : number of FiberTrace
   */
  template< typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel >
  PTR(FiberTrace< ImageT, MaskT, VarianceT >) makeNormFlatFiberTrace( PTR( const afwImage::MaskedImage< ImageT, MaskT, VarianceT >) const& maskedImage,
                                                                      PTR( const ::pfs::drp::stella::FiberTraceFunction ) const& fiberTraceFunctionWide,
                                                                      PTR( const ::pfs::drp::stella::FiberTraceFunctionControl ) const& fiberTraceFunctionControlNarrow,
                                                                      PTR( const ::pfs::drp::stella::FiberTraceProfileFittingControl ) const& fiberTraceProfileFittingControl,
                                                                      ImageT minSNR = 100.,
                                                                      std::size_t iTrace = 0 );

  /**
   * @brief: assign trace number to set of FiberTraces from x and y center by comparing the center position to the center positions of the zemax model
   * @param fiberTraceSet: FiberTraceSet to assign iTrace to
   * @param traceIds: shape(nfibers * nRows)
   * @param xCenters: shape(nfibers * nRows)
   */
  template< typename ImageT, typename MaskT, typename VarianceT, typename T, typename U, int I >
  void assignITrace( FiberTraceSet< ImageT, MaskT, VarianceT > & fiberTraceSet,
                     ndarray::Array< T, 1, I > const& traceIds,
                     ndarray::Array< U, 1, I > const& xCenters );

  /**
   * @brief: compare x and y center of fiberTrace to xCenters and yCenters to identify traceID
   * @param fiberTrace: fiber trace to identify
   * @param xCenters: shape(nfibers * nRows)
   * @param nTraces: number of fiber traces on CCD
   * @param nRows: number of CCD rows
   * @param startPos: fiber number to start searching
   * @return fiber trace number
   */
  template< typename ImageT, typename MaskT, typename VarianceT, typename U, int I >
  std::size_t findITrace( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                     ndarray::Array< U, 1, I > const& xCenters,
                     std::size_t nTraces,
                     std::size_t nRows,
                     std::size_t startPos = 0 );

    /**
     * @brief: add FiberTrace representation image to CCD image
     * @param fiberTrace: FiberTrace to mark in maskedImage's Mask
     * @param fiberTraceRepresentation: FiberTrace image to copy to ccdArray
     * @param ccdArray: Array to add the FiberTraceRepresentation to
     */
    template< typename ImageT, typename MaskT, typename VarianceT, typename arrayT, typename ccdImageT, int dim >
    void addFiberTraceToCcdArray( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                  afwImage::Image< arrayT > const& fiberTraceRepresentation,
                                  ndarray::Array< ccdImageT, 2, dim > & ccdArray );

    /**
     * @brief: add FiberTrace representation image to CCD image
     *         (wrapper for addFiberTraceToCcdImage(FiberTrace, Image, Array) until Swig can successfully parse
     *         a numpy array
     * @param fiberTrace : FiberTrace to mark in maskedImage's Mask
     * @param fiberTraceRepresentation : FiberTrace image to copy to ccdArray
     * @param ccdImage : Image to add the FiberTraceRepresentation to
     */
    template< typename ImageT, typename MaskT, typename VarianceT, typename arrayT, typename ccdImageT >
    void addFiberTraceToCcdImage( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                  afwImage::Image< arrayT > const& fiberTraceRepresentation,
                                  afwImage::Image< ccdImageT > & ccdImage );
     
    /**
     * @brief: Add one array into certain positions in another array
     *         The purpose of the function is to add a curved FiberTrace
     *         representation into an array representing a CCD image.
     * @param smallArr in: Array to be added to bigArr in the area defined by xMinMax and yMin
     * @param xMinMax in: 2D array of shape(smallArr.getShape()[0],2) containing the x limits where to add
     *                 each row from smallArr in
     * @param yMin in: row number in which to start adding smallArr into
     * @param bigArr in/out: Array to which to add smallArr
     */
    template< typename smallT, typename bigT, int I, int J >
    void addArrayIntoArray( ndarray::Array< smallT, 2, I > const& smallArr,
                            ndarray::Array< std::size_t, 2, 1 > const& xMinMax,
                            std::size_t const& yMin,
                            ndarray::Array< bigT, 2, J > & bigArr );

    /**
     * @brief Convert CCD coordinates into Trace coordinates
     * @param ccdCoordinates in: CCD Coordinates with the centre of the pixel (0.0, 0.0)
     * @param fiberTrace in: FiberTrace for which to make the coordinate conversion
     * output Coordinates : x (column) and y (row) in FiberTrace.image Coordinates
     */
    template<typename CoordT, typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
    dataXY<CoordT> ccdToFiberTraceCoordinates(
        dataXY<CoordT> const& ccdCoordinates,
        pfs::drp::stella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace);

    /**
     * @brief Convert FiberTrace coordinates into coordinates relative to the given CCD Coordinates
     * @param fiberTraceCoordinatesTrace in: Coordinates in the Trace Coordinate System
     * @param ccdCoordinatesCenter in: CCD Coordinates as center for new Coordinate system
     * @param fiberTrace in: FiberTrace for which to make the coordinate conversion
     * @param psf in: PSF for which to make the coordinate conversion
     * output Coordinates : x (column) and y (row) relative to the ccdCoordinatesCenter
     */
    template<typename CoordT, typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
    dataXY<CoordT> fiberTraceCoordinatesRelativeTo(
        dataXY<CoordT> const& fiberTraceCoordinates,
        dataXY<CoordT> const& ccdCoordinatesCenter,
        pfs::drp::stella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace
    );
}   

namespace utils{
    /**
     * @brief return raw pointer to ptr
     * @param ptr : object to which to return the raw pointer
     */
    template<typename T>
    const T* getRawPointer(const PTR(const T) & ptr);

     /**
      * @brief mark FiberTrace pixels in Mask image
      * @param fiberTrace : FiberTrace to mark in maskedImage's Mask
      * @param mask : mask to mark the FiberTrace in
      * @param value : value to Or into the FiberTrace mask
      */
     template< typename ImageT, typename MaskT, typename VarianceT >
     void markFiberTraceInMask( PTR( FiberTrace< ImageT, MaskT, VarianceT > ) const& fiberTrace,
                                PTR( afwImage::Mask< MaskT > ) const& mask,
                                MaskT value = 1);
  
}

}}}
#endif
