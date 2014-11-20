///TODO: replace integral in SlitFunc with point and IntegralNormalise with sumNormalise
///TODO: split profile calculation and 1d extraction

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
#include "Controls.h"
#include "math/Math.h"
#include "utils/Utils.h"
#include "cmpfit-1.2/MyFit.h"
#include "spline.h"
#include "PSF.h"
//#include "kriging/geostat.h"

#define stringify( name ) # name

//#define __DEBUG_BANDSOL__
#define __DEBUG_CALC2DPSF__
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
#define DEBUGDIR "/home/azuri/spectra/pfs/2014-11-02/debug/"// /home/azuri/entwicklung/idl/REDUCE/16_03_2013/"//stella/ses-pipeline/c/msimulateskysubtraction/data/"//spectra/elaina/eso_archive/red_564/red_r/"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

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
    explicit FiberTrace(
      unsigned int width, unsigned int height, unsigned int iTrace=0
    );

    explicit FiberTrace(
      afwGeom::Extent2I const & dimensions=afwGeom::Extent2I(), unsigned int iTrace=0
    );

    explicit FiberTrace(
      PTR(MaskedImageT) const &maskedImage, unsigned int iTrace=0
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

    /// Create _trace from maskedImage and _fiberTraceFunction
    /// Pre: _xCenters set/calculated
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

    /// Return _fiberTraceExtractionControl
    PTR(TwoDPSFControl) getTwoDPSFControl() const { return _twoDPSFControl; }

    /// Set the _fiberTraceExtractionControl
    bool setTwoDPSFControl(PTR(TwoDPSFControl) twoDPSFControl);// { _fiberTraceExtractionControl = fiberTraceExtractionControl; }

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
    
    bool isProfileSet() const {return _isProfileSet;}

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

    bool fitSpline(const blitz::Array<double, 2> &fiberTraceSwath_In,/// 1 bin of CCD (FiberTrace::Image)
                   const blitz::Array<int, 1> &iFirst_In,/// as calculated in SlitFunc
                   const blitz::Array<double, 1> &xOverSampled_In,/// see XVecArr in SlitFunc
                   blitz::Array<double, 1> &profileOverSampled_Out,/// output oversampled spatial profile
                   const blitz::Array<double, 2> &profileXValuesPerRowOverSampled_In,/// (i + 0.5) / double(overSample_In) - 1. + xCentersPixelFraction_In(i)
                   const blitz::Array<double, 1> &profileXValuesAllRows_In,/// i + 0.5 + (1. / (2. * overSample))
                   blitz::Array<double, 2> &profilePerRow_Out);/// output 2D profile image

    bool calculate2dPSFPerBin();
//    bool calculate2dPSF(const int yLow_In,
//                        const int yHigh_In,
//                        blitz::Array<double, 2> &PSF2D_Out);

    bool calculateSwathWidth_NBins_BinHeight_BinBoundY(int &swathWidth,
                                                       int &nBins,
                                                       int &binHeight,
                                                       blitz::Array<int, 2> &binBoundY);

    std::vector<PTR(pfs::drp::stella::PSF<ImageT, MaskT, VarianceT>)> getPSFVector() {return _psfVector;}
    
    PTR(pfs::drp::stella::PSF<ImageT, MaskT, VarianceT>) getPSF(int pos) {return _psfVector[pos];}
    
    void setITrace(unsigned int iTrace){_iTrace = iTrace;}
    unsigned int getITrace(){return _iTrace;}
    
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
    std::vector<PTR(pfs::drp::stella::PSF<ImageT, MaskT, VarianceT>)> _psfVector;
    unsigned int _iTrace;
//    unsigned int _iBin;
    //    std::vector<float> _fittedSky;
    bool _isXCentersCalculated;
//    bool _isImageSet;
    bool _isTraceSet;
    bool _isProfileSet;
    bool _isFiberTraceFunctionSet;
    bool _isFiberTraceExtractionControlSet;
    bool _isTwoDPSFControlSet;
    bool _isSpectrumExtracted;
    pfs::drp::stella::FiberTraceFunction _fiberTraceFunction;
    PTR(pfs::drp::stella::FiberTraceExtractionControl) _fiberTraceExtractionControl;
    PTR(pfs::drp::stella::TwoDPSFControl) _twoDPSFControl;


  protected:
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

    bool setTwoDPSFControl(TwoDPSFControl &twoDPSFControl);

    /// set profiles of all traces in this FiberTraceSet to respective FiberTraces in input set
    /// NOTE: the FiberTraces should be sorted by their xCenters before performing this operation!
    bool setAllProfiles(FiberTraceSet<ImageT, MaskT, VarianceT> &fiberTraceSet);

    /// re-order the traces in _traces by the xCenter of each trace
    void sortTracesByXCenter();

    /// calculate spatial profile and extract to 1D
    bool extractTraceNumber(int traceNumber);
    bool extractAllTraces();

    /// extract 1D spectrum from previously provided profile
    bool extractTraceNumberFromProfile(int traceNumber);
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
  
}

//  template<typename ImageT, typename MaskT, typename VarianceT>
//  PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) getShared(afwImage::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage);

}}}
int main();
#endif
