///TODO: split profile calculation and 1d extraction
///TODO: Create own class for FiberTraceProfile?
///TODO: Add deep option to copy constructors
#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include <iostream>
#include <utility>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions/Exception.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
//#include "blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "Controls.h"
#include "math/Math.h"
#include "math/CurveFitting.h"
//#include "math/MathBlitz.h"
#include "utils/Utils.h"
//#include "utils/UtilsBlitz.h"
#include "cmpfit-1.2/MPFitting_ndarray.h"
#include "spline.h"
#include "Spectra.h"

#define stringify( name ) # name

//#define __DEBUG_BANDSOL__
//#define __DEBUG_CALC2DPSF__
//#define __DEBUG_CALCPROFILE__
//#define __DEBUG_CALCPROFILESWATH__
//#define __DEBUG_CALCSWATHBOUNDY__
//#define __DEBUG_CHECK_INDICES__
//#define __DEBUG_CREATEFIBERTRACE__
//#define __DEBUG_EXTRACTFROMPROFILE__
//#define __DEBUG_FINDANDTRACE__
//#define __DEBUG_FIT__
//#define __DEBUG_SPLINE__
//#define __DEBUG_INTERPOL__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_MKPROFIM__
//#define __DEBUG_MKSLITFUNC__
//#define __DEBUG_SETFIBERTRACEFUNCTION__
//#define __DEBUG_SLITFUNC__
//#define __DEBUG_SLITFUNC_N__
//#define __DEBUG_SLITFUNC_PISKUNOV__
//#define __DEBUG_SLITFUNC_X__
//#define __DEBUG_TELLURIC__
//#define __DEBUG_TRACEFUNC__
//#define __DEBUG_UNIQ__
//#define __DEBUG_XCENTERS__
#define DEBUGDIR "/Users/azuri/spectra/pfs/2014-11-02/debug/"// /home/azuri/entwicklung/idl/REDUCE/16_03_2013/"//stella/ses-pipeline/c/msimulateskysubtraction/data/"//spectra/elaina/eso_archive/red_564/red_r/"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;

using namespace std;
namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a single fiber trace
 */
template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
class FiberTrace {
  public:
    typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

    // Class Constructors and Destructor
    explicit FiberTrace(size_t width = 0, size_t height = 0, size_t iTrace = 0);

    explicit FiberTrace(PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) const& maskedImage, 
                        PTR(const FiberTraceFunction) const& fiberTraceFunction, 
                        ndarray::Array<float const, 1, 1> const& xCenters,
                        size_t iTrace=0);
    
    explicit FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> &fiberTrace, bool const deep=false);
    
    virtual ~FiberTrace() {}

    /// Return shared pointer to the 2D MaskedImage of this fiber trace
    PTR(MaskedImageT) getTrace() { return _trace; }
    const PTR(const MaskedImageT) getTrace() const { return _trace; }
    
    /// Set the 2D image of this fiber trace to imageTrace
    /// Pre: _fiberTraceFunction must be set
    bool setTrace(PTR(MaskedImageT) & trace);// { _trace = trace; }

    /// Return the pointer to the image of this fiber trace
    PTR(afwImage::Image<ImageT>) getImage() const { return _trace->getImage(); }

    /// Set the image pointer of this fiber trace to image
    bool setImage(const PTR(afwImage::Image<ImageT>) &image);// { _trace->getImage() = image; }

    /// Return the pointer to the mask of this fiber trace
    PTR(afwImage::Mask<MaskT>) getMask() const{ return _trace->getMask(); }

    /// Set the mask pointer of this fiber trace to mask
    bool setMask(const PTR(afwImage::Mask<MaskT>) &mask);// { _trace->getMask() = mask; }

    /// Return the pointer to the variance of this fiber trace
    PTR(afwImage::Image<VarianceT>) getVariance() const { return _trace->getVariance(); }

    /// Set the variance pointer of this fiber trace to variance
    bool setVariance(const PTR(afwImage::Image<VarianceT>) &variance);// { _trace->getVariance() = variance; }

    /// Return the image of the spatial profile
    PTR(afwImage::Image<float>) getProfile() const{ return _profile; }

    /// Set the _profile of this fiber trace to profile
    bool setProfile(const PTR(afwImage::Image<float>) &profile);

    /// Extract the spectrum of this fiber trace using the _profile
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractFromProfile();
    
    /// Simple Sum Extraction of this fiber trace
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractSum();

    /// Create _trace from maskedImage and _fiberTraceFunction
    /// Pre: _xCenters set/calculated
    bool createTrace(const PTR(const MaskedImageT) & maskedImage);

    /// Return _fiberTraceFunction
    const PTR(const FiberTraceFunction) getFiberTraceFunction() const { return _fiberTraceFunction; }

    /// Return _fiberTraceProfileFittingControl
    PTR(FiberTraceProfileFittingControl) getFiberTraceProfileFittingControl() const { return _fiberTraceProfileFittingControl; }

    /// Set the _fiberTraceProfileFittingControl
    bool setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);// { _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl; }
    
    /// Return the x-centers of the fiber trace
    const ndarray::Array<float const, 1, 1> getXCenters() const { return _xCenters; }

    /// Set the x-center of the fiber trace
    /// Pre: _fiberTraceFunction must be set
//    bool setXCenters(const PTR(std::vector<float>) &xCenters);// { _xCenters = xCenters; }

    /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
    PTR(afwImage::Image<float>) getReconstructed2DSpectrum(const Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum) const;

    /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
    PTR(afwImage::Image<float>) getReconstructedBackground(const Spectrum<ImageT, MaskT, VarianceT, VarianceT> & backgroundSpectrum) const;

    /// Return shared pointer to an image containing the reconstructed 2D spectrum + background of the FiberTrace
    PTR(afwImage::Image<float>) getReconstructed2DSpectrum(const Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum,
                                                           const Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const;
    
    bool calcProfile();
    ndarray::Array<float, 2, 2> calcProfileSwath(ndarray::Array<ImageT const, 2, 2> const& imageSwath,
                                                 ndarray::Array<MaskT const, 2, 2> const& maskSwath,
                                                 ndarray::Array<VarianceT const, 2, 2> const& varianceSwath,
                                                 ndarray::Array<float const, 1, 1> const& xCentersSwath,
                                                 size_t const iSwath);

/*    bool fitSpline(const blitz::Array<double, 2> &fiberTraceSwath_In,/// 1 bin of CCD (FiberTrace::Image)
                   const blitz::Array<double, 1> &xOverSampled_In,/// see XVecArr in SlitFunc
                   blitz::Array<double, 1> &profileOverSampled_Out,/// output oversampled spatial profile
                   const blitz::Array<double, 2> &profileXValuesPerRowOverSampled_In,/// (i + 0.5) / double(overSample_In) - 1. + xCentersPixelFraction_In(i)
                   const blitz::Array<double, 1> &profileXValuesAllRows_In,/// i + 0.5 + (1. / (2. * overSample))
                   blitz::Array<double, 2> &profilePerRow_Out);/// output 2D profile image
*/
    ndarray::Array<size_t, 2, 1> calcSwathBoundY(const size_t swathWidth_In) const;
    
    void setITrace(const size_t iTrace){_iTrace = iTrace;}
    size_t getITrace() const {return _iTrace;}
    bool isTraceSet() const {return _isTraceSet;}
    bool isProfileSet() const {return _isProfileSet;}
    bool isFiberTraceProfileFittingControlSet() const {return _isFiberTraceProfileFittingControlSet;}
    size_t getWidth() const {return _trace->getImage()->getWidth();}
    size_t getHeight() const {return _trace->getImage()->getHeight();}
    ndarray::Array<double, 1, 1> getTraceCoefficients() const;
//    bool setTraceCoefficients(ndarray::Array<double, 1, 1> const& coeffs);
    PTR(FiberTrace) getPointer();

    std::vector<PTR(std::vector<float>)> _overSampledProfileFitXPerSwath;
    std::vector<PTR(std::vector<float>)> _overSampledProfileFitYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputXMeanPerSwath;
    std::vector<PTR(std::vector<float>)> _profileFittingInputYMeanPerSwath;
    
  private:
    ///TODO: replace variables with smart pointers?????
    PTR(MaskedImageT) _trace;
    PTR(afwImage::Image<float>) _profile;
    const ndarray::Array<float const, 1, 1> _xCenters;
    size_t _iTrace;
    bool _isTraceSet;
    bool _isProfileSet;
    bool _isFiberTraceProfileFittingControlSet;
    const PTR(const FiberTraceFunction) _fiberTraceFunction;
    PTR(FiberTraceProfileFittingControl) _fiberTraceProfileFittingControl;

    /// for debugging purposes only

  protected:
};

/************************************************************************************************************/
/**
 * \brief Describe a set of fiber traces
 *
 */
template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
class FiberTraceSet {
  public:
    typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

    /// Class Constructors and Destructor
    
    /// Creates a new FiberTraceSet object of size nTraces
    explicit FiberTraceSet(size_t nTraces=0);

    /// Copy constructor
    /// If fiberTraceSet is not empty, the object shares ownership of fiberTraceSet's fiber trace vector and increases the use count.
    /// If fiberTraceSet is empty, an empty object is constructed (as if default-constructed).
    explicit FiberTraceSet(const FiberTraceSet &fiberTraceSet, bool const deep = false);
    
    /// Construct an object with a copy of fiberTraceVector
///    explicit FiberTraceSet(const std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> &fiberTraceVector)
///        : _traces(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(fiberTraceVector))
///        {}
        
    virtual ~FiberTraceSet() {}

    /// Return the number of apertures
    size_t size() const { return _traces->size(); }
    
    /// Extract FiberTraces from new MaskedImage
    bool createTraces(const PTR(const MaskedImageT) &maskedImage);

    /// Return the FiberTrace for the ith aperture
    PTR(FiberTrace<ImageT, MaskT, VarianceT>) &getFiberTrace(const size_t i);

    PTR(FiberTrace<ImageT, MaskT, VarianceT>) const& getFiberTrace(const size_t i) const;
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const size_t iStart, const size_t iEnd=0);

    /// Set the ith FiberTrace
    bool setFiberTrace(const size_t i,     ///< which aperture?
                       const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace ///< the FiberTrace for the ith aperture
                      );

    /// Add one FiberTrace to the set
    bool addFiberTrace(const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace, const size_t iTrace = 0);

    PTR(std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) getTraces() const { return _traces; }
//    PTR(const std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) getTraces() const { return _traces; }

    bool setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);

    /// set profiles of all traces in this FiberTraceSet to respective FiberTraces in input set
    /// NOTE: the FiberTraces should be sorted by their xCenters before performing this operation!
    bool setAllProfiles(const PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) &fiberTraceSet);

    /// re-order the traces in _traces by the xCenter of each trace
    void sortTracesByXCenter();

    /// calculate spatial profile and extract to 1D
//    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractTraceNumber(const size_t traceNumber);
//    PTR(SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) extractAllTraces();

    ///TODO:
    /// Extract spectrum and background for one slit spectrum
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(Spectrum<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground(const size_t traceNumber)

    ///TODO:
    /// Extract spectrum and background for all slit spectra
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(SpectrumSet<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground()

    /// calculate profiles for all traces
    bool calcProfileAllTraces();
    
    /// extract 1D spectrum from previously provided profile
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractTraceNumberFromProfile(const size_t traceNumber);
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
    PTR(std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) _traces; // traces for each aperture
};

namespace math{
  /** 
   * * identifies and traces the fiberTraces in maskedImage, and extracts them into individual FiberTraces
   * * FiberTraces in returned FiberTraceSet will be sorted by their xCenter positions
   * Set I_NTermsGaussFit to
   *       1 to look for maximum only without GaussFit
   *       3 to fit Gaussian
   *       4 to fit Gaussian plus constant (sky)
   *         Spatial profile must be at least 5 pixels wide
   *       5 to fit Gaussian plus linear term (sloped sky)
   *         Spatial profile must be at least 6 pixels wide
   * NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
   **/
  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
  PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                     const PTR(const FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl);
  
  /**
   * @brief: returns ndarray containing the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
   *         NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
   */
  ndarray::Array<float, 1, 1> calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                size_t const& ccdHeightIn = 0,
                                                size_t const& ccdWidthIn = 0);
  ndarray::Array<float, 1, 1> calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                ndarray::Array<float, 1, 1> const& yIn,
                                                size_t const& ccdHeightIn = 0,
                                                size_t const& ccdWidthIn = 0);

}

namespace utils{
  template<typename T>
  const T* getRawPointer(const PTR(const T) & ptr);

}

//  template<typename ImageT, typename MaskT, typename VarianceT>
//  PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) getShared(afwImage::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage);

}}}
//int main();
#endif
