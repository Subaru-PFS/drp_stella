///TODO: split profile calculation and 1d extraction
///TODO: Create own class for FiberTraceProfile?
///TODO: Add deep option to copy constructors
#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions/Exception.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "Controls.h"
#include "math/Math.h"
#include "utils/Utils.h"
#include "cmpfit-1.2/MyFit.h"
#include "spline.h"
#include "Spectra.h"

#define stringify( name ) # name

//#define __DEBUG_BANDSOL__
//#define __DEBUG_CALC2DPSF__
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
//#define __DEBUG_TRACEFUNC__
//#define __DEBUG_UNIQ__
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
//    typedef boost::shared_ptr<FiberTrace> Ptr;
//    typedef boost::shared_ptr<FiberTrace const> ConstPtr;

    // Class Constructors and Destructor
    explicit FiberTrace(unsigned int width = 0, unsigned int height = 0, unsigned int iTrace = 0);

 //   explicit FiberTrace(afwGeom::Extent2I const & dimensions, unsigned int iTrace = 0);

    explicit FiberTrace(PTR(const MaskedImageT) const& maskedImage, 
                        PTR(const FiberTraceFunction) const& fiberTraceFunction, 
                        PTR(const std::vector<float>) const& xCenters,
                        unsigned int iTrace=0);
    
    explicit FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> &fiberTrace);
    
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
    bool setFiberTraceProfileFittingControl(const PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);// { _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl; }

    /// Calculate the x-centers of the fiber trace
    //bool calculateXCenters();//FiberTraceFunctionControl const& fiberTraceFunctionControl);
    
    /// Return the x-centers of the fiber trace
    const PTR(const std::vector<float>) getXCenters() const { return _xCenters; }

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

    /**
     *        Methods from Piskunov and Valenti
     **/
    /**
     *       MkSlitFunc
     *       Make Slit Function
     *       Parameter(s) to be tuned:
     *         swath_width - swath width in columns
     *         SF_SMOOTH - smoothing accross dispersion
     *         SP_SMOOTH - smoothing in dispersion direction
     *         OSAMPLE   - slit function is reconstructed on
     *                     subpixel grid with stepsize
     *                     OSAMPLE times smaller than CCD
     *                     pixels. Larger OSAMPLE require
     *                     more computing time and larger
     *                     SF_SMOOTH. Plots of residuals
     *                     allow to verify if the OSAMPLE
     *                     is good enough.
     **/
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) MkSlitFunc();
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) MkSlitFunc(const blitz::Array<string, 1> &S_A1_Args,     //: in
                                     void *ArgV[]);                        //: in
    /* KeyWords and Values:  //    LAMBDA_SF   : double          : in
     *                       //    LAMBDA_SP   : int             : out
     *                       //      WING_SMOOTH_FACTOR = double  : in
     *                       //      SWATH_WIDTH : int             : in
     *                             FIBERTRACENUMBER
     *                             BLZ         : blitz::Array<double, 1>: out
     *                       //      MASK        : blitz::Array<double, 2>: in
     *                       //      CCD_GAIN    : double          : in
     *                       //      CCD_READN   : double          : in
     *                       //      NO_SCATTER  : void
     *                       //      TELLURIC    : int (0-none, 1-Piskunov, 2-mine, 3-mine with sky background)
     *                             FILENAME    : CString         : in
     *                       //      XCOR_PROF   : int             : in (Number of Cross-correlations of profile and spectrum from one pixel to the left to one pixel to the right)
     */

    /**
     *      SlitFunc
     *      Calculates slit function for one swath
     **/
    bool SlitFunc(const blitz::Array<double, 2> &D_A2_ImM,         ///: in
                  unsigned int maxIterSig_In,
                  const blitz::Array<double, 1> &xCentersPixelFraction_In, //: in
                  blitz::Array<double, 1> &D_A1_SP_Out,                     ///: out
                  blitz::Array<double, 2> &D_A2_SF_Out,                     ///: out
                  const blitz::Array<string, 1> &S_A1_Args,            ///: in
                  void *ArgV[]);                            ///: in
    /** KeyWords and Values:  NOISE      = double          : in
     *                        IM_OUT     = blitz::Array<double, 2>: out
     *                        PROF_OUT   = blitz::Array<double, 2>: out
     *                        USE_ROW    = int             : in
     *                        BAD        = blitz::Array<int, 1>   : out
     *                        MASK       = blitz::Array<double, 2>: in/out
     *                        STOP       = int [0,1]                          : in
     *                        SKY        = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
     *                        ERRORS     = blitz::Array<double, 2>(D_A2_ImM.rows(), D_A2_ImM.cols()): in/out
     *                        ERRORS_OUT = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
     *                        ERR_SKY    = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
     *                        SP_FIT     = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
     *                        I_BIN      = int
     *                        FIBERTRACENUMBER = unsigned int: in
     * from MkSliFunc:
     *         ///      if ((I_Pos = util::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
     *        ///        s_a1(pppos) = "FIBERTRACENUMBER";
     *        ///      if (fiberTraceProfileFittingControl_In.xCorProf > 0)
     *        ///        s_a1(pppos) = "XCOR_PROF";
     *        ///      s_a1(pppos) = "SP_OUT";
     *        ///      s_a1(pppos) = "STOP";
     *        ///      s_a1(pppos) = "MASK";
     *        //       if (fiberTraceProfileFittingControl_In.telluric > 1)
     *        //       {
     *        //         s_a1(pppos) = "SKY";
     *        //         pppos++;
     *        //         s_a1(pppos) = "SP_FIT";
     *        //         pppos++;
     *        //       }
     *        //    if (ErrorsRead){
     *        //      s_a1(pppos) = "ERRORS";
     *        //      pppos++;
     *        //      s_a1(pppos) = "ERRORS_OUT";
     *        //      pppos++;
     *        //      s_a1(pppos) = "ERRORS_SP_OUT";
     *        //      pppos++;
     *        //      if (I_Telluric > 1)
     *        //      {
     *        //        s_a1(pppos) = "ERR_SKY";
     *        //        pppos++;
     *        //      }
     *        //    }
     *        //    s_a1(pppos) = "I_BIN";
     *        //    pppos++;
     *        //    s_a1(pppos) = "DEBUGFILES_SUFFIX";
     *
     * in SlitFunc:
     *       //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "DEBUGFILES_SUFFIX");
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "I_BIN");
     *      //      Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERRORS");
     *      //      Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERRORS_OUT");
     *      //      Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERRORS_SP_OUT");
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SP_OUT");
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SFO_OUT");
     *      //    if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "XCOR_PROF")) >= 0)
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MASK")) >= 0)
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "PROF_OUT")) >= 0)
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SKY");
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERR_SKY");
     *      //    int Pos_Stop = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "STOP");
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "USE_ROWS")) >= 0)
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "NOISE");
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "IM_OUT")) >= 0)
     *      //    if ((Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "USE_ROW")) >= 0)// && TempIntB != 0)
     *      //    Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SP_FIT");
     *
     **/

    bool fitSpline(const blitz::Array<double, 2> &fiberTraceSwath_In,/// 1 bin of CCD (FiberTrace::Image)
                   const blitz::Array<int, 1> &iFirst_In,/// as calculated in SlitFunc
                   const blitz::Array<double, 1> &xOverSampled_In,/// see XVecArr in SlitFunc
                   blitz::Array<double, 1> &profileOverSampled_Out,/// output oversampled spatial profile
                   const blitz::Array<double, 2> &profileXValuesPerRowOverSampled_In,/// (i + 0.5) / double(overSample_In) - 1. + xCentersPixelFraction_In(i)
                   const blitz::Array<double, 1> &profileXValuesAllRows_In,/// i + 0.5 + (1. / (2. * overSample))
                   blitz::Array<double, 2> &profilePerRow_Out);/// output 2D profile image

    ndarray::Array<int, 2, 1> calculateBinBoundY(int swathWidth_In) const;
    
    void setITrace(unsigned int iTrace){_iTrace = iTrace;}
    unsigned int getITrace() const {return _iTrace;}
    bool isTraceSet() const {return _isTraceSet;}
    bool isProfileSet() const {return _isProfileSet;}
    bool isFiberTraceProfileFittingControlSet() const {return _isFiberTraceProfileFittingControlSet;}
    unsigned int getWidth() const {return _trace->getImage()->getWidth();}
    unsigned int getHeight() const {return _trace->getImage()->getHeight();}
    PTR(FiberTrace) getPointer();
    
  private:
    ///TODO: replace variables with smart pointers?????
    PTR(MaskedImageT) _trace;
    PTR(afwImage::Image<float>) _profile;
    const PTR(const std::vector<float>) _xCenters;
    unsigned int _iTrace;
    bool _isTraceSet;
    bool _isProfileSet;
    bool _isFiberTraceProfileFittingControlSet;
    const PTR(const FiberTraceFunction) _fiberTraceFunction;
    PTR(FiberTraceProfileFittingControl) _fiberTraceProfileFittingControl;


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
    explicit FiberTraceSet(unsigned int nTraces=0);

    /// Copy constructor
    /// If fiberTraceSet is not empty, the object shares ownership of fiberTraceSet's fiber trace vector and increases the use count.
    /// If fiberTraceSet is empty, an empty object is constructed (as if default-constructed).
    explicit FiberTraceSet(const FiberTraceSet &fiberTraceSet)
        : _traces(fiberTraceSet.getTraces())
        {}
    
    /// Construct an object with a copy of fiberTraceVector
///    explicit FiberTraceSet(const std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> &fiberTraceVector)
///        : _traces(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(fiberTraceVector))
///        {}
        
    virtual ~FiberTraceSet() {}

    /// Return the number of apertures
    int size() const { return _traces->size(); }
    
    /// Extract FiberTraces from new MaskedImage
    bool createTraces(const PTR(const MaskedImageT) &maskedImage);

    /// Return the FiberTrace for the ith aperture
    PTR(FiberTrace<ImageT, MaskT, VarianceT>) &getFiberTrace(const unsigned int i ///< desired aperture
                                                            );

    PTR(FiberTrace<ImageT, MaskT, VarianceT>) const& getFiberTrace(const unsigned int i ///< desired aperture
                                                                  ) const;
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const unsigned int iStart, const unsigned int iEnd=0);

    /// Set the ith FiberTrace
    bool setFiberTrace(const unsigned int i,     ///< which aperture?
                       const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace ///< the FiberTrace for the ith aperture
                      );

    /// Add one FiberTrace to the set
    bool addFiberTrace(const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace ///< the FiberTrace for the ith aperture
    );

    PTR(std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) getTraces() const { return _traces; }
//    PTR(const std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) getTraces() const { return _traces; }

    bool setFiberTraceProfileFittingControl(const PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl);

    /// set profiles of all traces in this FiberTraceSet to respective FiberTraces in input set
    /// NOTE: the FiberTraces should be sorted by their xCenters before performing this operation!
    bool setAllProfiles(const PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) &fiberTraceSet);

    /// re-order the traces in _traces by the xCenter of each trace
    void sortTracesByXCenter();

    /// calculate spatial profile and extract to 1D
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractTraceNumber(int traceNumber);
    PTR(SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) extractAllTraces();

    ///TODO:
    /// Extract spectrum and background for one slit spectrum
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(Spectrum<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground(int traceNumber)

    ///TODO:
    /// Extract spectrum and background for all slit spectra
    /// Returns vector of size 2 (0: Spectrum, 1: Background)
    /// PTR(std::vector<PTR(SpectrumSet<ImageT, MaskT, VarianceT, ImageT>)>) extractSpectrumAndBackground()
    
    /// extract 1D spectrum from previously provided profile
    PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) extractTraceNumberFromProfile(int traceNumber);
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
   *  NOTE: the center of a pixel is [0.,0.], so the lower left corner of a pixels is [-0.5,-0.5]
   **/
  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
  PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                     const PTR(const FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl);
  
  PTR(const std::vector<float>) calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunction,
                                                      unsigned int const& ccdHeight,
                                                      unsigned int const& ccdWidth);

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
