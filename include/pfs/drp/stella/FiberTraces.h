///TODO: replace integral in SlitFunc with point and IntegralNormalise with sumNormalise

#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
//#include <algorithm>
#include <iostream>
//#include <memory>
#include "lsst/base.h"
//#include "lsst/afw/geom/Box.h"
//#include "lsst/afw/image/Image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "cmpfit-1.2/MyFit.h"
#include "spline.h"

#define stringify( name ) # name

//#define __DEBUG_BANDSOL__
//#define __DEBUG_CREATEFIBERTRACE__
//#define __DEBUG_EXTRACTFROMPROFILE__
//#define __DEBUG_FINDANDTRACE__
//#define __DEBUG_FIT__
//#define __DEBUG_INTERPOL__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_MKPROFIM__
//#define __DEBUG_MKSLITFUNC__
//#define __DEBUG_SETFIBERTRACEFUNCTION__
//#define __DEBUG_SLITFUNC__
//#define __DEBUG_SLITFUNC_N__
//#define __DEBUG_SLITFUNC_PISKUNOV__
#define __DEBUG_SLITFUNC_X__
//#define __DEBUG_TRACEFUNC__
//#define __DEBUG_CHECK_INDICES__
#define DEBUGDIR "/home/azuri/spectra/pfs/2014-10-14/debug/"// /home/azuri/entwicklung/idl/REDUCE/16_03_2013/"//stella/ses-pipeline/c/msimulateskysubtraction/data/"//spectra/elaina/eso_archive/red_564/red_r/"

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a>b)?a:b)

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella {

/**
 * Description of fiber trace function
 */
struct FiberTraceFunctionControl {
  /// enum corresponding to legal values of interpolation string
  enum INTERPOLATION {  CHEBYSHEV=0, LEGENDRE, CUBIC, LINEAR, POLYNOMIAL, NVALUES };
  std::vector<std::string> INTERPOLATION_NAMES = { stringify( CHEBYSHEV ),
                                                   stringify( LEGENDRE ),
                                                   stringify( CUBIC ),
                                                   stringify( LINEAR ),
                                                   stringify( POLYNOMIAL) };
  LSST_CONTROL_FIELD(interpolation, std::string, "Interpolation schemes");
  LSST_CONTROL_FIELD(order, unsigned int, "Polynomial order");
  LSST_CONTROL_FIELD(xLow, float, "Lower (left) limit of aperture relative to center position of trace in x (< 0.)");
  LSST_CONTROL_FIELD(xHigh, float, "Upper (right) limit of aperture relative to center position of trace in x");
  
  FiberTraceFunctionControl() :
      interpolation("POLYNOMIAL"),
      order(3),
      xLow(-4.),
      xHigh(4.) {}
};

struct FiberTraceFunction {
  FiberTraceFunctionControl fiberTraceFunctionControl; /// User defined Polynomial interpolation and order, xLow, xHigh (width of fiber trace)
  float xCenter; /// Central position of fiber trace in x
  unsigned int yCenter; /// Central position of fiber trace in y
  int yLow; /// lower limit of fiber trace relative to center (< 0)
  unsigned int yHigh; /// lower limit of fiber trace relative to center (>= 0)
  std::vector<double> coefficients; /// polynomial coefficients of fiber trace function
  
  FiberTraceFunction() :
  fiberTraceFunctionControl(),
  xCenter(0.),
  yCenter(0),
  yLow(0),
  yHigh(0),
  coefficients(4) {}
};

struct FiberTraceFunctionFindingControl {
  /// enum corresponding to legal values of interpolation string
  LSST_CONTROL_FIELD(fiberTraceFunctionControl, FiberTraceFunctionControl, "Interpolation function and order");
  LSST_CONTROL_FIELD(apertureFWHM, float, "FWHM of an assumed Gaussian spatial profile for tracing the spectra");
  LSST_CONTROL_FIELD(signalThreshold, float, "Signal below this threshold is assumed zero for tracing the spectra");   // Should we use lsst::afw::detection::Threshold?
  LSST_CONTROL_FIELD(nTermsGaussFit, unsigned short, "1 to look for maximum only without GaussFit; 3 to fit Gaussian; 4 to fit Gaussian plus constant (sky), Spatial profile must be at least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky), Spatial profile must be at least 6 pixels wide");
  LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
  LSST_CONTROL_FIELD(minLength, unsigned int, "Minimum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(maxLength, unsigned int, "Maximum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(nLost, unsigned int, "Number of consecutive times the trace is lost before aborting the traceing");
  
  FiberTraceFunctionFindingControl() :
  fiberTraceFunctionControl(),
  apertureFWHM(2.5),
  signalThreshold(0.),
  nTermsGaussFit(3),
  saturationLevel(65000.),
  minLength(10),
  maxLength(4096),
  nLost(10)
  {}
};
  
/**
 * Control Fiber trace extraction
 */
struct FiberTraceExtractionControl {
    enum {  NONE=0, BEFORE_EXTRACTION, DURING_EXTRACTION, NVALUES } TELLURIC;/// Determine background/sky not at all or before or during profile determination/extraction
    std::vector<std::string> TELLURIC_NAMES = { stringify( NONE ),
                                                stringify( BEFORE_EXTRACTION ),
                                                stringify( DURING_EXTRACTION ) };
    LSST_CONTROL_FIELD(ccdReadOutNoise, float, "CCD readout noise");
    LSST_CONTROL_FIELD(swathWidth, unsigned int, "Size of individual extraction swaths");
    LSST_CONTROL_FIELD(telluric, std::string, "Method for determining the background (+sky in case of slit spectra, default: NONE)");
    LSST_CONTROL_FIELD(overSample, unsigned int, "Oversampling factor for the determination of the spatial profile (default: 10)");
    LSST_CONTROL_FIELD(maxIterSF, unsigned int, "Maximum number of iterations for the determination of the spatial profile (default: 8)");
    LSST_CONTROL_FIELD(maxIterSky, unsigned int, "Maximum number of iterations for the determination of the (constant) background/sky (default: 10)");
    LSST_CONTROL_FIELD(maxIterSig, unsigned int, "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)");
    LSST_CONTROL_FIELD(lambdaSF, float, "Lambda smoothing factor for spatial profile (default: 1. / overSample)");
    LSST_CONTROL_FIELD(lambdaSP, float, "Lambda smoothing factor for spectrum (default: 0)");
    LSST_CONTROL_FIELD(wingSmoothFactor, float, "Lambda smoothing factor to remove possible oscillation of the wings of the spatial profile (default: 0.)");
    LSST_CONTROL_FIELD(xCorProf, unsigned short, "Number of Cross-correlations of profile and spectrum from one pixel to the left to one pixel to the right");
    
    FiberTraceExtractionControl() :
        ccdReadOutNoise(1.),
        swathWidth(500),
        telluric("NONE"),
        overSample(15),
        maxIterSF(8),
        maxIterSky(0),
        maxIterSig(1),
        lambdaSF(1./static_cast<float>(overSample)),
        lambdaSP(0.),
        wingSmoothFactor(2.),
        xCorProf(0) {}

    FiberTraceExtractionControl(FiberTraceExtractionControl &fiberTraceExtractionControl) :
        ccdReadOutNoise(fiberTraceExtractionControl.ccdReadOutNoise),
        swathWidth(fiberTraceExtractionControl.swathWidth),
        telluric(fiberTraceExtractionControl.telluric),
        overSample(fiberTraceExtractionControl.overSample),
        maxIterSF(fiberTraceExtractionControl.maxIterSF),
        maxIterSky(fiberTraceExtractionControl.maxIterSky),
        maxIterSig(fiberTraceExtractionControl.maxIterSig),
        lambdaSF(fiberTraceExtractionControl.lambdaSF),
        lambdaSP(fiberTraceExtractionControl.lambdaSP),
        wingSmoothFactor(fiberTraceExtractionControl.wingSmoothFactor),
        xCorProf(fiberTraceExtractionControl.xCorProf) {}
};

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
    explicit FiberTrace(
      unsigned int width, unsigned int height
    );
  
    explicit FiberTrace(
      afwGeom::Extent2I const & dimensions=afwGeom::Extent2I()
    );

    explicit FiberTrace(
      PTR(MaskedImageT) const &maskedImage
    );
    
    virtual ~FiberTrace() {}

    /// Return the 2D image of this fiber trace
    MaskedImageT getTrace() { return _trace; }
    
    /// Set the 2D image of this fiber trace to imageTrace
    bool setTrace( MaskedImageT & trace);// { _trace = trace; }
    
    /// Return the pointer to the image of this fiber trace
    PTR(afwImage::Image<ImageT>) getImage() { return _trace.getImage(); }
    
    /// Set the image pointer of this fiber trace to image
    bool setImage( PTR(afwImage::Image<ImageT>) image);// { _trace.getImage() = image; }
    
    /// Return the pointer to the mask of this fiber trace
    PTR(afwImage::Mask<MaskT>) getMask() { return _trace.getMask(); }
    
    /// Set the mask pointer of this fiber trace to mask
    bool setMask( PTR(afwImage::Mask<MaskT>) mask);// { _trace.getMask() = mask; }
    
    /// Return the pointer to the variance of this fiber trace
    PTR(afwImage::Image<VarianceT>) getVariance() { return _trace.getVariance(); }
    
    /// Set the variance pointer of this fiber trace to variance
    bool setVariance( PTR(afwImage::Image<VarianceT>) variance);// { _trace.getVariance() = variance; }

    /// Return the image of the spatial profile
    PTR(afwImage::Image<float>) getProfile(){ return _profile; }

    /// Set the _profile of this fiber trace to profile
    bool setProfile(PTR(afwImage::Image<float>) profile);
    
    /// Extract the spectrum of this fiber trace using the _profile
    bool extractFromProfile();
    bool extractFromProfile(const blitz::Array<string, 1> &S_A1_Args,     //: in
                            void *ArgV[]);                        //: in
    
    /// Create _trace from _image and _fiberTraceFunction
    /// Pre: _xCenters set/calculated
    /// Side Effects: resets _profile to afwImage<float>(_trace.getDimensions())
    bool createTrace(PTR(MaskedImageT) const & maskedImage);
        
    /// Return the masked CCD image
//    PTR(MaskedImageT) getMaskedImage() { return _maskedImage; }
    
    /// Set the masked CCD image to maskedImage
//    void setMaskedImage(const PTR(MaskedImageT) & maskedImage);

    /// Return _fiberTraceFunction
    FiberTraceFunction getFiberTraceFunction() const { return _fiberTraceFunction; }

    /// Set the _fiberTraceFunction
    bool setFiberTraceFunction(const FiberTraceFunction &fiberTraceFunction);// { _fiberTraceFunction = fiberTraceFunction; }
    
    /// Return _fiberTraceExtractionControl
    PTR(FiberTraceExtractionControl) getFiberTraceExtractionControl() const { return _fiberTraceExtractionControl; }
    
    /// Set the _fiberTraceExtractionControl
    bool setFiberTraceExtractionControl(PTR(FiberTraceExtractionControl) fiberTraceExtractionControl);// { _fiberTraceExtractionControl = fiberTraceExtractionControl; }
    
    /// Calculate the x-centers of the fiber trace
    bool calculateXCenters();//FiberTraceFunctionControl const& fiberTraceFunctionControl);
    
    /// Return the extracted spectrum of the fiber trace
    std::vector<float> getSpectrum() const { return _spectrum; }
    
    /// Return the extracted background of the fiber trace
    std::vector<float> getBackground() const { return _background; }
    
    /// Return the x-centers of the fiber trace
    std::vector<float> getXCenters() const { return _xCenters; }
    
    /// Set the x-center of the fiber trace
    /// Pre: _fiberTraceFunction must be set
    bool setXCenters(const std::vector<float> &xCenters);// { _xCenters = xCenters; }
    
    /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
    afwImage::Image<float> getReconstructed2DSpectrum() const; 
    
    /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
    afwImage::Image<float> getReconstructedBackground() const; 
    
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
    bool MkSlitFunc();
    bool MkSlitFunc(const blitz::Array<string, 1> &S_A1_Args,     //: in
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
                  const blitz::Array<double, 1> &D_A1_XCenters_In, //: in
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
     *         ///      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
     *        ///        s_a1(pppos) = "FIBERTRACENUMBER";
     *        ///      if (fiberTraceExtractionControl_In.xCorProf > 0)
     *        ///        s_a1(pppos) = "XCOR_PROF";
     *        ///      s_a1(pppos) = "SP_OUT";
     *        ///      s_a1(pppos) = "STOP";
     *        ///      s_a1(pppos) = "MASK";
     *        //       if (fiberTraceExtractionControl_In.telluric > 1)
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
    
    bool fitSpline(const blitz::Array<double, 2> &fiberTraceSwath,
                   //const blitz::Array<double, 1> &xCentersPixelFraction,
                   const blitz::Array<double, 1> &iFirst,
                   blitz::Array<double, 2> &profile);
    
private:
    ///TODO: replace variables with smart pointers?????
    MaskedImageT _trace;
    PTR(afwImage::Image<float>) _profile;
    ///TODO: remove _ccdWidth and _ccdHeight and put as input parameters into calculateXCenters()
    int _ccdWidth;
    int _ccdHeight;
    std::vector<float> _xCenters;
    std::vector<float> _spectrum;
    std::vector<float> _spectrumVariance;
    std::vector<float> _background;
    std::vector<float> _backgroundVariance;
    //    std::vector<float> _fittedSky;
    bool _isXCentersCalculated;
//    bool _isImageSet;
    bool _isTraceSet;
    bool _isProfileSet;
    bool _isFiberTraceFunctionSet;
    bool _isFiberTraceExtractionControlSet;
    bool _isSpectrumExtracted;
    FiberTraceFunction _fiberTraceFunction;
    PTR(FiberTraceExtractionControl) _fiberTraceExtractionControl;
};

/************************************************************************************************************/
/**
 * \brief Describe a set of fiber traces
 *
 * \note If a profArray is passed to the ctor, you are required to pass ImageFiberTraces to setFiberTrace
 */
template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
class FiberTraceSet {
  public:
    typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
  
    // Class Constructors and Destructor
    explicit FiberTraceSet(unsigned int nTraces=0)
        : _traces(nTraces)
        {}

    virtual ~FiberTraceSet() {}
    
    /// Return the number of apertures
    int size() const { return _traces.size(); }
    
    /// Return the FiberTrace for the ith aperture
    FiberTrace<ImageT, MaskT, VarianceT> &getFiberTrace(int const i ///< desired aperture
                             ) { return *_traces.at(i); }
                             
    FiberTrace<ImageT, MaskT, VarianceT> const& getFiberTrace(int const i ///< desired aperture
                                   ) const { return *_traces.at(i); }
                                   
    /// Set the ith FiberTrace
    bool setFiberTrace(int const i,     ///< which aperture?
                       PTR(FiberTrace<ImageT, MaskT, VarianceT>) trace ///< the FiberTrace for the ith aperture
                      );
    
    /// Set the ith FiberTrace
    void addFiberTrace(PTR(FiberTrace<ImageT, MaskT, VarianceT>) trace ///< the FiberTrace for the ith aperture
    );
    
    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> & getTraces(){ return _traces; }
    
    bool setFiberTraceExtractionControl(FiberTraceExtractionControl &fiberTraceExtractionControl);
    
    /// set profiles of all traces in this FiberTraceSet to respective FiberTraces in input set
    /// NOTE: the FiberTraces should be sorted by their xCenters before performing this operation!
    bool setAllProfiles(FiberTraceSet<ImageT, MaskT, VarianceT> &fiberTraceSet);
    
    /// re-order the traces in _traces by the xCenter of each trace
    void sortTracesByXCenter();
    
    /// calculate spatial profile and extract to 1D
    bool extractTraceNumber(const unsigned int traceNumber);
    bool extractAllTraces();

    /// extract 1D spectrum from previously provided profile
    bool extractTraceNumberFromProfile(const unsigned int traceNumber);
    bool extractAllTracesFromProfile();
    
  private:
    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> _traces; // traces for each aperture
};

  namespace math{

    /** Set I_NTermsGaussFit to
     *       1 to look for maximum only without GaussFit
     *       3 to fit Gaussian
     *       4 to fit Gaussian plus constant (sky)
     *         Spatial profile must be at least 5 pixels wide
     *       5 to fit Gaussian plus linear term (sloped sky)
     *         Spatial profile must be at least 6 pixels wide
     *    template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
     *    PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
     **/
    template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
    FiberTraceSet<ImageT, MaskT, VarianceT> findAndTraceApertures(const PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                       const pfs::drp::stella::FiberTraceFunctionFindingControl &fiberTraceFunctionFindingControl);
    
    
    /*****************************************************************/
    /*  Sub method for CubicSpline, Legendre, and Chebyshev          */
    /*****************************************************************/
    double GetNormalized(double XVal,
                         double XMin,
                         double XMax);
                         
    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetA(double XVal,
                double XMin,
                double XMax,
                int Order);
                
    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetB(double XVal,
                double XMin,
                double XMax,
                int Order);
                
    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    long GetJ(double XVal,
              double XMin,
              double XMax,
              int Order);
              
    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetS(double XVal,
                double XMin,
                double XMax,
                int Order);

    bool LinearSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                      const blitz::Array<double, 1> &D_A1_Coeffs_In,
                      double D_XCenter_In,
                      double D_YCenter_In,
                      double D_YMin_In,
                      double D_YMax_In,
                      double D_Low_In,
                      double D_High_In,
                      int I_Order_In,
                      int I_NCols_In,
                      int I_NRows_In);
    
    bool CubicSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In,
                     int I_NRows_In);
    

    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function, 
     *  i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, and given values yP1 and 
     *  yPN for the first derivative of the interpolating function at points 1 and 
     *  N, respectively, this routine returns an Array y2(0:N-1) that contains the 
     *  second derivatives of the interpolating function at the tabulated points 
     *  x_i. If yP1 and/or yPN are equal to 1x10^30 or larger, the routine is 
     *  signaled to set the corresponding boundary condition for a natural spline, 
     *  with zero second derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In, 
                const blitz::Array<double, 1> &y_In, 
                double yP1, 
                double yPN, 
                blitz::Array<double, 1> &y_Out);

    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function, 
     *  i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, this routine returns an 
     *  Array y2(0:N-1) that contains the second derivatives of the interpolating 
     *  function at the tabulated points x_i. The routine is signaled to set the 
     *  corresponding boundary condition for a natural spline, with zero second 
     *  derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In, 
                const blitz::Array<double, 1> &y_In, 
                blitz::Array<double, 1> &y_Out);

    /**
     *  SplInt
     *  Given the Arrays xVec_In(0:N-1) and y1_In(0:N-1), which tabulate a 
     *  function (whith the xVec_In(i)'s in order), and given the array y2_In(0:N-1), 
     *  which is the output from Spline above, and given a value of x_In, this 
     *  routine returns a cubic-spline interpolated value y_Out;
     **/
    bool SplInt(const blitz::Array<double, 1> &xVec_In, 
                blitz::Array<double, 1> &y1_In, 
                blitz::Array<double, 1> &y2_In, 
                double x_In, 
                double *y_Out);
    
    bool ChebyLegend(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     double &D_YMin_Out,
                     double &D_YMax_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In,
                     int I_NRows_In,
                     const string &S_Function_In);
    
    bool Legendre(blitz::Array<double, 1> &D_A1_XCenters_Out,
                  double &D_YMin_Out,
                  double &D_YMax_Out,
                  const blitz::Array<double, 1> &D_A1_Coeffs_In,
                  double D_XCenter_In,
                  double D_YCenter_In,
                  double D_YMin_In,
                  double D_YMax_In,
                  double D_XLow_In,
                  double D_XHigh_In,
                  int I_Order_In,
                  int I_NCols_In,
                  const int I_NRows_In);
    
    bool Chebyshev(blitz::Array<double, 1> &D_A1_XCenters_Out,
                   double &D_YMin_Out,
                   double &D_YMax_Out,
                   const blitz::Array<double, 1> &D_A1_Coeffs_In,
                   double D_XCenter_In,
                   double D_YCenter_In,
                   double D_YMin_In,
                   double D_YMax_In,
                   double D_XLow_In,
                   double D_XHigh_In,
                   int I_Order_In,
                   int I_NCols_In,
                   const int I_NRows_In);
    
    /*************************************************************
     * Poly
     * 
     * INPUTS:
     *       D_A1_X_In:      The variable.  1D array.
     *
     *       D_A1_Coeffs_In: The 1D array of polynomial coefficients.  The degree of
     *                       of the polynomial is N_ELEMENTS(VecCoeffs) - 1.
     *
     * OUTPUTS:
     *       POLY returns a result equal to:
     *                C[0] + c[1] * X + c[2]*X^2 + ...
     *
    **/
    blitz::Array<double, 1> Poly(const blitz::Array<double, 1> &D_A1_X_In,
                                 const blitz::Array<double, 1> &D_A1_Coeffs_In);
    
    double Poly(const double D_X_In,
                const blitz::Array<double, 1> &D_A1_Coeffs_In);
    
    /**
     *        Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
     **/
    blitz::Array<int,1>* GetIndex(const blitz::Array<int,1> &I_A1_Where, 
                                  int &I_NInd_Out);
    
    /**
     *      Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
     **/
    bool GetIndex(const blitz::Array<int,1> &I_A1_Where, 
                  int &I_NInd_Out, 
                  blitz::Array<int, 1> &I_IndArr_Out);
    
    /**
     *      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
     *      blitz::Array<int, 2> *P_I_A2_Out(I_NInd_Out, 2)
     **/
    blitz::Array<int,2>* GetIndex(const blitz::Array<int,2> &I_A2_Where, 
                                  int &I_NInd_Out);
    
    /**
     *      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
     *      blitz::Array<int, 2> I_IndArr_Out(I_NInd_Out, 2)
     **/
    bool GetIndex(const blitz::Array<int,2> &I_A2_Where, 
                  int &I_NInd_Out, 
                  blitz::Array<int, 2> &I_IndArr_Out);

    /**
     *      ValueLocate
     *      Returns the successive indices of the Range of the two indices of the monotonically increasing or decreasing Vector vec_In, 
     *        in which Val falls.
     *      If Vector is monotonically increasing, the result is
     *      if j = -1       valueVec_In(i) < vec_In(0)
     *      if 0 <= j < N-1 vec_In(j) <= valueVec_In(i) < vec_In(j+1)
     *      if j = N-1      vec_In(N-1) <= valueVec_In(i)
     * 
     *      If Vector is monotonically decreasing, the result is
     *      if j = -1       vec_In(0) <= valueVec_In(i)
     *      if 0 <= j < N-1 vec_In(j+1) <= valueVec_In(i) < vec_In(j)
     *      if j = N-1      valueVec_In(i) < vec_In(N-1)
     **/
    blitz::Array<int, 1>* valueLocate(const blitz::Array<double, 1> &vec_In,
                                      const blitz::Array<double, 1> &valueVec_In);
                                      
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to I_A2_MinCenMax_Out
     **/
    bool calcMinCenMax(const blitz::Array<float, 1> &xCenters_In,
                       float xHigh_In,
                       float xLow_In,
                       int yCenter_In,
                       int yLow_In,
                       int yHigh_In,
                       int nPixCutLeft_In,
                       int nPixCutRight_In,
                       blitz::Array<int, 2> &I_A2_MinCenMax_Out);

    /**
     * Calculates Slit Function for each pixel in an aperture row from oversampled Slit Function D_A1_OSF_In,
     * and writes result to D_A1_SF_Out
     **
    bool CalcSF(const blitz::Array<double, 1> &xCenters_In,
                unsigned int I_Row_In,
                float xHigh_In,
                float xLow_In,
                unsigned int overSample_In,
                const blitz::Array<double, 1> &D_A1_OSF_In,
                const pfs::drp::stella::FiberTrace<float>::FiberTrace::MaskedImageT &image_In,
                blitz::Array<double, 1> &D_A1_SF_Out);*/
    
    /**
     * Fix(double)
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater than or equal to D_In is returned,
     * e.g. D_In = -99.8 => returns -99.
     **/
    template <typename T> 
    int Fix(T D_In);
    //%template(fixd) Fix(double);
    
    /**
      Fix(blitz::Array<double, 1> &VecArr)
      Returns an Array of the same size containing the Fix integer values of VecArr.
    **/
    template <typename T> 
    blitz::Array<int, 1> Fix(const blitz::Array<T, 1> &VecArr);
                              
    /**
     Fix(blitz::Array<double, 2> &Arr)
     Returns an Array of the same size containing the Fix integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<int, 2> Fix(const blitz::Array<T, 2> &Arr);
    
    /**
     * FixL(double)
     * Returns integer value cut at decimal point (See int Fix(double)).
     **/
    template <typename T> 
    long FixL(T D_In);
    
    /**
      FixL(blitz::Array<double, 1> &VecArr)
      Returns an Array of the same size containing the fix long integer values of VecArr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 1> FixL(const blitz::Array<T, 1> &VecArr);

    /**
     FixL(blitz::Array<double, 2> &Arr, CString Mode)
     Returns an Array of the same size containing the long integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 2> FixL(const blitz::Array<T, 2> &Arr);
        
    /**
     * Int(double)
     * Returns integer portion of D_In. If D_In is negative returns the first negative integer less than or equal to Number,
     * e.g. D_In = -99.8 => returns -100.
     **/
    template <typename T> 
    int Int(T D_In);
    
    /**
     *      Fix(blitz::Array<double, 1> &VecArr)
     *      Returns an Array of the same size containing the Int integer values of VecArr.
     **/
    template <typename T> 
    blitz::Array<int, 1> Int(const blitz::Array<T, 1> &VecArr);
    
    /**
     *     Fix(blitz::Array<double, 2> &Arr)
     *     Returns an Array of the same size containing the Int integer values of Arr (see int Int(double D_In)).
     **/
    template <typename T> 
    blitz::Array<int, 2> Int(const blitz::Array<T, 2> &Arr);
    
    /**
     * Returns integer value cut at decimal point (See int Int(double)).
     **/
    template <typename T> 
    long Long(T D_In);
    
    /**
     *      Returns an Array of the same size containing the Int long integer values of VecArr (see int Int(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 1> Long(const blitz::Array<T, 1> &VecArr);
    
    /**
     *     Returns an Array of the same size containing the long integer values of Arr (see int Int(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 2> Long(const blitz::Array<T, 2> &Arr);
    
    /**
     *      Returns an Array of the same size containing the float values of VecArr.
     **/
    template <typename T> 
    blitz::Array<float, 1> Float(const blitz::Array<T, 1> &VecArr);
    
    /**
     *      Returns an Array of the same size containing the float values of VecArr.
     **/
    template <typename T> 
    void Float(const blitz::Array<T, 1> &VecArr, blitz::Array<float, 1> &VecArr_Out);
    
    /**
     *     Returns an Array of the same size containing the float values of Arr (see int Int(double D_In)).
     **/
    template <typename T> 
    blitz::Array<float, 2> Float(const blitz::Array<T, 2> &Arr);
    
    /**
     *     Returns an Array of the same size containing the float values of Arr (see int Int(double D_In)).
     **/
    template <typename T> 
    void Float(const blitz::Array<T, 2> &Arr, blitz::Array<float, 2> &Arr_Out);
    
    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T> 
    void Double(const blitz::Array<T, 1> &Arr, blitz::Array<double, 1> &Arr_Out);
    
    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T> 
    blitz::Array<double, 1> Double(const blitz::Array<T, 1> &Arr);
    
    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T> 
    void Double(const blitz::Array<T, 2> &Arr, blitz::Array<double, 2> &Arr_Out);
    
    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T> 
    blitz::Array<double, 2> Double(const blitz::Array<T, 2> &Arr);
    
    template <typename T>
    int Round(const T ToRound);
    
    template <typename T>
    T Round(const T ToRound, int DigitsBehindDot);
    
    template <typename T> 
    long RoundL(const T ToRound);
    
    /**
     *      Replicate(double val, int Len);
     *      Out: blitz::Array<double, 1>(Len)
     **/
    template<typename T>
    blitz::Array<T, 1> Replicate(T val, int Len);
        
    /**
     * Calculate Integral under curve from D_A1_XInt(0) to D_A1_XInt(1)
     **/
    bool IntegralUnderCurve(const blitz::Array<double, 1> &D_A1_XIn,
                            const blitz::Array<double, 1> &D_A1_YIn,
                            const blitz::Array<double, 1> &D_A1_XInt,
                            double &D_Integral_Out);
    
    /**
     * Calculate Integral under line between two points
     * D_A2_Coords_In(0,0) = x0
     * D_A2_Coords_In(0,1) = y0
     * D_A2_Coords_In(1,0) = x1
     * D_A2_Coords_In(1,1) = y1
     * **/
    bool IntegralUnderLine(const blitz::Array<double, 2> &D_A2_Coords_In,
                           double &D_Integral_Out);
    
    /**
     * Integral-normalise a function
     **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           const blitz::Array<double, 1> &D_A1_YIn,
                           blitz::Array<double, 1> &D_A1_YOut);
                           
    /**
     * Integral-normalise a function
     **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           blitz::Array<double, 1> &D_A1_YInOut);
                                                  
    /**
     * PURPOSE:
     *   Perform a least-square polynomial fit with optional error estimates.
     *
     *   This routine uses matrix inversion.  A newer version of this routine,
     *   SVDFIT, uses Singular Value Decomposition.  The SVD technique is more
     *   flexible, but slower.
     *
     * INPUTS:
     *   X:  The independent variable vector.
     *
     *   Y:  The dependent variable vector, should be same length as x.
     *
     *   Degree: The degree of the polynomial to fit.
     *
     * OUTPUTS:
     *   POLY_FIT returns a vector of coefficients with a length of NDegree+1.
     *
     * KEYWORDS:
     *   CHISQ=chisq: double: out:   
     *     Sum of squared errors divided by MEASURE_ERRORS if specified.
     *
     *   COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out:   
     *     Covariance matrix of the coefficients.
     *
     *   MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in: 
     *     Set this keyword to a vector containing standard
     *     measurement errors for each point Y[i].  This vector must be the same
     *     length as X and Y.
     *
     *     Note - For Gaussian errors (e.g. instrumental uncertainties),
     *       MEASURE_ERRORS should be set to the standard
     *       deviations of each point in Y. For Poisson or statistical weighting
     *       MEASURE_ERRORS should be set to sqrt(Y).
     *
     *   SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out:  
     *     The 1-sigma error estimates of the returned parameters.
     *
     *     Note: if MEASURE_ERRORS is omitted, then you are assuming that
     *       your model is correct. In this case, SIGMA is multiplied
     *       by SQRT(CHISQ/(N-M)), where N is the number of points
     *       in X and M is the number of terms in the fitting function.
     *       See section 15.2 of Numerical Recipes in C (2nd ed) for details.
     *
     *   STATUS=status: int: out:
     *     Set this keyword to a named variable to receive the status
     *     of the operation. Possible status values are:
     *     0 for successful completion, 1 for a singular array (which
     *     indicates that the inversion is invalid), and 2 which is a
     *     warning that a small pivot element was used and that significant
     *     accuracy was probably lost.
     * 
     *   YFIT:   blitz::Vector of calculated Y's. These values have an error
     *           of + or - YBAND.
     *
    CHISQ=chisq: double: out
    COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out
    MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in
    SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out
    STATUS=status: int: out
    YFIT=yfit: blitz::Array<double, 1>(D_A1_X_In.size()): out
    **/
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 int I_Degree,
                 const blitz::Array<string, 1> &S_A1_Args,
                 void *ArgV[],
                 blitz::Array<double, 1>* P_D_A1_Out);

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 int I_Degree,
                 blitz::Array<double, 1>* P_D_A1_Out);
    
/** Additional Keywords:
    REJECTED=blitz::Array<int, 1>
    NOT_REJECTED=blitz::Array<int, 1>
    N_REJECTED=int
    **/
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree,
                 double D_Reject,
                 const blitz::Array<string, 1> &S_A1_Args,
                 void *ArgV[],
                 blitz::Array<double, 1>* P_D_A1_Out);
    
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree,
                 double D_LReject,
                 double D_HReject,
                 unsigned int I_NIter,
                 const blitz::Array<string, 1> &S_A1_Args,
                 void *ArgV[],
                 blitz::Array<double, 1>* P_D_A1_Out);

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree,
                 double D_Reject,
                 blitz::Array<double, 1>* P_D_A1_Out);
    
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree,
                 double D_LReject,
                 double D_HReject,
                 unsigned int I_NIter,
                 blitz::Array<double, 1>* P_D_A1_Out);
    
    
    /**
     *  Creates float array containing the index numbers as values
     **/
    blitz::Array<float, 1> FIndGenArr(int len);
    
    /**
     *  Creates double array containing the index numbers as values
     **/
    blitz::Array<double, 1> DIndGenArr(int len);
    
    /**
     *  Creates long array containing the index numbers as values
     **/
    blitz::Array<long, 1> LIndGenArr(int len);
    
    /**
     *  Creates int array containing the index numbers as values
     **/
    blitz::Array<int, 1> IndGenArr(int len);
    
    bool removeSubArrayFromArray(blitz::Array<int, 1> &A1_Array_InOut, 
                                  const blitz::Array<int, 1> &A1_SubArray);
  
     
    /**
      PURPOSE:
               Linearly interpolate vectors with a regular or irregular grid.
               Quadratic or a 4 point least-square fit to a quadratic
               interpolation may be used as an option.

      INPUTS:
             V:      The input vector can be any type except string.

         For regular grids:
             N:      The number of points in the result when both
                     input and output grids are regular.

         Irregular grids:
             X:      The absicissae values for V.  This vector must
                     have same # of elements as V.  The values MUST be
                     monotonically ascending or descending.

             U:      The absicissae values for the result.  The result
                     will have the same number of elements as U.  U
                     does not need to be monotonic.  If U is outside
                     the range of X, then the closest two endpoints of
                     (X,V) are linearly extrapolated.

      Keyword Input Parameters:
             LSQUADRATIC = if set, interpolate using a least squares
               quadratic fit to the equation y = a + bx + cx^2, for
               each 4 point neighborhood (x[i-1], x[i], x[i+1], x[i+2])
               surrounding the interval, x[i] <= u < x[i+1].

             QUADRATIC = if set, interpolate by fitting a quadratic
                         y = a + bx + cx^2, to the three point neighborhood
                         (x[i-1], x[i], x[i+1]) surrounding the interval
                         x[i] <= u < x[i+1].

             SPLINE = if set, interpolate by fitting a cubic spline to
                      the 4 point neighborhood (x[i-1], x[i], x[i+1],
                      x[i+2]) surrounding the interval, x[i] <= u <
                      x[i+1].

             Note: if LSQUADRATIC or QUADRATIC or SPLINE is not set,
                   the default linear interpolation is used.

      OUTPUTS:
             INTERPOL returns a floating-point vector of N points
             determined by interpolating the input vector by the
             specified method.

   PROCEDURE:
             For linear interpolation,
               Result(i) = V(x) + (x - FIX(x)) * (V(x+1) - V(x))
               where   x = i*(m-1)/(N-1) for regular grids.
                       m = # of elements in V, i=0 to N-1.

             For irregular grids, x = U(i).
                       m = number of points of input vector.

             For QUADRATIC interpolation, the equation y=a+bx+cx^2 is
               solved explicitly for each three point interval, and is
               then evaluated at the interpolate.

             For LSQUADRATIC interpolation, the coefficients a, b,
               and c, from the above equation are found, for the four
               point interval surrounding the interpolate using a
               least square fit.  Then the equation is evaluated at
               the interpolate.
               For SPLINE interpolation, a cubic spline is fit over
               the 4 point interval surrounding each interpolate,
               using the routine SPL_INTERP().
       **/
    /**
     InterPol linear, not regular
    **/
    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  blitz::Array<double, 1> &out);

    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  blitz::Array<double, 1> &out,
                  bool preserveFlux);

    /**
      InterPol
       The InterPol function performs linear, quadratic, or spline interpolation on vectors with an irregular grid.
     **/
    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &out);

    /**
      InterPol
       This function performs linear, quadratic, or spline interpolation on vectors with a regular grid.
     **
    bool InterPol(blitz::Array<double, 1> &v,
                  long n,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &out);*/

    /**
      HInterPol
      Help function for InterPol methods
     **/
    bool HInterPol(const blitz::Array<double, 1> &v,
                   const blitz::Array<double, 1> &x,
                   blitz::Array<int, 1> &s,
                   const blitz::Array<double, 1> &u,
                   const blitz::Array<string, 1> &keyWords_In,
                   blitz::Array<double,1> &out);
 
    
    /**
     *        LsToFit
     **/
    bool LsToFit(const blitz::Array<double, 1> &XXVecArr, 
                 const blitz::Array<double, 1> &YVecArr, 
                 double XM, 
                 double &D_Out);
    
    /**
     *        InvertGaussJ(AArray, BArray)
     *        FROM: Numerical Recipes
     *        Linear equation solution by Gauss-Jordan elimination
     *        AArray(0:N-1, 0:N-1) is the input matrix. BArray(0:N-1,
     *        0:M-1) is input containing the m right-hand side vectors.
     *        On output, AArray is replaced by its matrix inverse, and
     *        BArray is replaced by the corresponding set of solution
     *        vectors.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray, 
                      blitz::Array<double, 2> &BArray);
    
    /**
     *        InvertGaussJ(AArray)
     *        FROM: Numerical Recipes
     *        Linear equation solution by Gauss-Jordan elimination with B == Unity
     *        AArray(0:N-1, 0:N-1) is the input matrix.
     *        On output, AArray is replaced by its matrix inverse.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray);
    
    /**
     *      MatrixATimesB(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 2>(A.rows(), B.cols())
     **/
    blitz::Array<double, 2>* MatrixATimesB(const blitz::Array<double, 2> &A, 
                                           const blitz::Array<double, 2> &B);
    
    /**
     *      MatrixBTimesA(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 2>(B.rows(), A.cols())
     **/
    blitz::Array<double, 2>* MatrixBTimesA(const blitz::Array<double, 2> &A,
                                           const blitz::Array<double, 2> &B);
    
    /**
     *      MatrixTimesVecArr(blitz::Array<double, 2> &Arr, blitz::Array<double, 1> &B);
     *      Out: blitz::Array<double, 1>(A.rows())
     **/
    blitz::Array<double, 1>* MatrixTimesVecArr(const blitz::Array<double, 2> &A, 
                                               const blitz::Array<double, 1> &B);
    
    /**
     *      VecArrTimesMatrix(blitz::Array<double, 1> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 1>(B.cols())
     *      equivalent to IDL::operator #
     *      computes array elements by multiplying the rows of the first array by the columns of the second array
     **/
    blitz::Array<double, 1>* VecArrTimesMatrix(const blitz::Array<double, 1> &A, 
                                               const blitz::Array<double, 2> &B);
    
    /**
     *      VecArrACrossB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     *      Out: blitz::Array<double, 2>(A.size(), B.size())
     **/
    blitz::Array<double, 2>* VecArrACrossB(const blitz::Array<double, 1> &A, 
                                           const blitz::Array<double, 1> &B);
    
    /**
     *      VecArrACrossB(blitz::Array<int, 1> &Arr, blitz::Array<int, 1> &B);
     *      Out: blitz::Array<int, 2>(A.size(), B.size())
     **/
    blitz::Array<int, 2>* VecArrACrossB(const blitz::Array<int, 1> &A, 
                                        const blitz::Array<int, 1> &B);
    
    /**
     *      VecArrAScalarB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     *      Out: double
     **/
    double VecArrAScalarB(const blitz::Array<double, 1> &A, 
                          const blitz::Array<double, 1> &B);
    
    /**
     *      Reform(blitz::Array<double, 1> &Arr, int DimA, int DimB);
     *      Reformats blitz::Vector to Array of given size
     **/
    template<typename T>
    blitz::Array<T, 2>* Reform(const blitz::Array<T, 1> &Arr, 
                               int DimA, 
                               int DimB);
    
    /**
     *      Reform(blitz::Array<double, 2> &Arr);
     *      Reformates an Array to a blitz::Vector
     **/
    template<typename T>
    blitz::Array<T, 1>* Reform(const blitz::Array<T, 2> &Arr);
    
    /**
     *        GetSubArrCopy(blitz::Array<double, 1> &DA1_In, blitz::Array<int, 1> &IA1_Indices, blitz::Array<double, 1> &DA1_Out) const
     *        Copies the values of DA1_In(IA1_Indices) to DA1_Out
     **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 1> &DA1_In,
                       const blitz::Array<int, 1> &IA1_Indices,
                       blitz::Array<T, 1> &DA1_Out);
    
    /**
     *        GetSubArrCopy(blitz::Array<double, 2> &DA2_In, blitz::Array<int, 1> &IA1_Indices, int I_Mode_In, blitz::Array<double, 2> &DA2_Out) const
     *        Copies the values of DA1_In(IA1_Indices) to DA2_Out
     *        I_Mode_In: 0: IA1_Indices are row numbers
     *                   1: IA1_Indices are column numbers
     **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 2> &DA2_In,
                       const blitz::Array<int, 1> &IA1_Indices,
                       int I_Mode_In,
                       blitz::Array<T, 2> &DA2_Out);
    
    /**
     *        GetSubArrCopy(blitz::Array<double, 2> &DA1_In, blitz::Array<int, 3> &I_A3_Indices) const
     *        Copies the values of D_A2_In(I_A3_Indices(row,col,0), I_A3_Indices(row,col,1)) to D_A2_Out
     **/
    template<typename T>
    blitz::Array<T, 2> GetSubArrCopy(const blitz::Array<T, 2> &D_A2_In,
                                      const blitz::Array<int, 3> &I_A3_Indices);
                                       
    /**
     *        function CountPixGTZero
     *        replaces input vector with vector of the same size where values are zero where the input vector is lower than
     *        or equal to zero and all other values represent the number of gt-zero values since the last zero value
     **/
    template<typename T>
    bool CountPixGTZero(blitz::Array<T, 1> &vec_InOut);
    
    /**
     *        function FirstIndexWithValueGEFrom
     *        returns first index of integer input vector where value is greater than or equal to I_MinValue, starting at index I_FromIndex
     *        returns -1 if values are always smaller than I_MinValue
     **/
    template<typename T>
    int FirstIndexWithValueGEFrom(const blitz::Array<T, 1> &vecIn, const T minValue, const int fromIndex);
    
    /**
     *        function LastIndexWithZeroValueBefore
     *        returns last index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 before I_StartPos
     **/
    template<typename T>
    int LastIndexWithZeroValueBefore(const blitz::Array<T, 1> &vec_In, const int startPos_In);
    
    /**
     *        function FirstIndexWithZeroValueFrom
     *        returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 past I_StartPos
     **/
    template<typename T>
    int FirstIndexWithZeroValueFrom(const blitz::Array<T, 1> &vec_In, const int startPos_In);
    
    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// y: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// x: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         bool B_WithSky,                           /// with sky: in
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[]);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1

    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// y: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// x: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[]);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1
     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0.
              * D_SP_Out must be >= 0.
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
      bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      ///: in
                           const blitz::Array<double, 2> &D_A2_SF_In,       ///: in
                           blitz::Array<double,1> &D_A1_SP_Out,             ///: out
                           blitz::Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                           void *ArgV_In[]);                   ///: in/out
  /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out

      bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      ///: in
                           const blitz::Array<double, 2> &D_A2_SF_In,       ///: in
                           blitz::Array<double,1> &D_A1_SP_Out,             ///: out
                           blitz::Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           bool B_WithSky,                           ///: with sky: in
                           const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                           void *ArgV_In[]);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    
    /**
     *       Helper function to calculate incomplete Gamma Function
     **/
    double GammLn(double D_X_In);
    
    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    bool GCF(double *P_D_Gamser_In, double a, double x, double *P_D_GLn);
    
    /**
     *      Function to calculate incomplete Gamma Function P
     **/
    bool GammP(double a, double x, double *D_Out);
    
    /**
     *      Function to calculate incomplete Gamma Function Q = 1 - P
     **/
    bool GammQ(double a, double x, double *D_Out);
    
    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    bool GSER(double *P_D_Gamser_In, double a, double x, double *P_D_GLn);
    
    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr);
    
    template<typename T>
    T Median(const blitz::Array<T, 2> &Arr, bool B_IgnoreZeros);
    
    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr, 
             const blitz::Array<string, 1> &S_A1_Args_In, 
             void *PP_Args[]);
    
//    template<typename T>
//    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &arr, 
//                                  int Width);
    
    template<typename T>
    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &arr, 
                                  int Width, 
                                  const std::string &Mode = std::string("NORMAL"));

    template<typename T>
    T Select(const blitz::Array<T, 1> &arr, int KThSmallest);

    bool IsOddNumber(long No);
    
    template<typename T>
    blitz::Array<T, 1> BubbleSort(const blitz::Array<T, 1> &T_A1_ArrIn);

    /**
     *      SortIndices(blitz::Array<double, 1> D_A1_In)
     *      Returns an integer array of the same size like <D_A1_In>,
     *      containing the indixes of <D_A1_In> in sorted order.
     **/
    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In);
    
    /**
     *       function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
     *       task: Returns Row specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In);
    
    /**
     *       function GetColFromIndex(int I_Index_In, int I_NRows_In) const
     *       task: Returns Col specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     **/
    int GetColFromIndex(int I_Index_In, int I_NRows_In);
    
    /**
     *      BandSol(blitz::Array<double, 2> &D_A2_A_In, blitz::Array<double, 1> &D_A1_R_In, int N, int I_ND) const
     * 
     *      bandsol solve a sparse system of linear equations with
     *      band-diagonal matrix.
     *      Band is assumed to be symmetrix relative to the main diaginal.
     *      Usage:
     *      bandsol(D_A2_A_In, D_A1_R_In, I_N, I_ND)
     *      where D_A2_A_In is 2D array [I_N,m] where I_N - is the number
     *      of equations and I_ND is the width of the band (3 for
     *      tri-diagonal system), I_ND is always an odd number. The main
     *      diagonal should be in D_A2_A_In(*,I_ND/2)
     *      The first lower subdiagonal should be in D_A2_A_In(1:I_N-1,I_ND-2-1),
     *      the first upper subdiagonal is in D_A2_A_In(0:I_N-2,I_ND/2+1)
     *      etc. For example:
     *                    / 0 0 X X X \
     *                    | 0 X X X X |
     *                    | X X X X X |
     *                    | X X X X X |
     *      D_A2_A_In =   | X X X X X |
     *                    | X X X X X |
     *                    | X X X X X |
     *                    | X X X X 0 |
     *                    \ X X X 0 0 /
     *      D_A1_R_In is the array of RHS of size I_N.
     *      Argv: blitz::Array<double, 2> &D_A2_A_InOut, blitz::Array<double, 1> &D_A1_R_InOut, int N, int I_ND
     **/
    void BandSol(int Argc, void *Argv[]);
    
    /**
     *       bool TriDag
     *       Solves for a vector blitz::Array<double, N> UVecArr the tridiagonal
     *       linear set given by equation
     * [ b_1  c_1  0  ...                       ]   [  u_1  ]   [  r_1  ]
     * [ a_2  b_2  c_2 ...                      ]   [  u_2  ]   [  r_2  ]
     * [            ...                         ] * [  ...  ] = [  ...  ]
     * [            ...  a_(N-1) b_(N-1) c_(N-1)]   [u_(N-1)]   [r_(N-1)]
     * [            ...     0     a_N      b_N  ]   [  u_N  ]   [  r_N  ]
     *      BVecArr, CVecArr, and RVecArr are input vectors and are not
     *      modified.
     **/
    bool TriDag(blitz::Array<double, 1> &AVecArr, 
                blitz::Array<double, 1> &BVecArr, 
                blitz::Array<double, 1> &CVecArr, 
                blitz::Array<double, 1> &RVecArr, 
                blitz::Array<double, 1> &UVecArr);
    /**
     *        NAME:
     *            UNIQ
     * 
     *        PURPOSE:
     *            Return the subscripts of the unique elements in an array.
     * 
     *            This command is inspired by the Unix uniq(1) command.
     * 
     *        CATEGORY:
     *            Array manipulation.
     * 
     *        CALLING SEQUENCE:
     *            Uniq(blitz::Array<int, 1> IA1_In, blitz::Array<int, 1> IA1_Out)
     * 
     *        INPUTS:
     *            blitz::Array<int, 1> IA1_In:  The array to be scanned. The number of dimensions of the array is not important.  The array must be sorted into monotonic order.
     * 
     *        OUTPUTS:
     *            An array of indicies into ARRAY (blitz::Array<int, 1> IA1_Out) is returned.  The expression:
     * 
     *            Uniq(IA1_In, IA1_Out);
     *            blitz::Array<int, 1> SubArr(this->GetSubArr(IA1_In, I_A1_Out))
     * 
     *        will be a copy of the sorted Array with duplicate adjacent elements removed.
     * 
     **/
    bool Uniq(const blitz::Array<int, 1> &IA1_In,
              blitz::Array<int, 1> &IA1_Out);
    
//    template<typename T>
//    void resize(blitz::Array<T, 1> &arr_InOut, unsigned int newSize);
  }/// end namespace math
  
  namespace utils{
    
    /**
     *       Returns Position of <str_In> in Array of strings <S_A1_In>, if <S_A1_In> contains string <str_In>, else returns -1.
     **/
    int KeyWord_Set(const blitz::Array<string, 1> &S_A1_In, 
                    const string &str_In);
    
    template<typename T>
    bool WriteFits(const blitz::Array<T,2>* image_In, const string &fileName_In);
    
    template<typename T>
    bool WriteFits(const blitz::Array<T,1>* image_In, const string &fileName_In);

    /**
      *      task: Writes Array <I_A1_In> to file <CS_FileName_In>
      *      CS_Mode: [binary, ascii]
      **/
    template<typename T, int N>
    bool WriteArrayToFile(const blitz::Array<T, N> &I_A1_In,
                          const string &S_FileName_In,
                          const string &S_Mode);

    /**
             task: Writes Array <D_A2_In> to file <CS_FileName_In>
             CS_Mode: [binary, ascii]
     **/
//    template<typename T, int N>
//    bool WriteArrayToFile(const blitz::Array<T, 2> &D_A2_In,
//                          const string &S_FileName_In,
//                          const string &S_Mode);

  }

}}}
int main();
#endif
