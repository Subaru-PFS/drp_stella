///TODO: calculate2dPSF: remove outliers in PSF

#ifndef __PFS_DRP_STELLA_PSF_H__
#define __PFS_DRP_STELLA_PSF_H__

#include <iostream>
#include <limits>
#include <vector>

#include "cmpfit-1.2/MPFitting_ndarray.h"
#include "Controls.h"
#include "FiberTraces.h"
#include "lsst/afw/image.h"
#include "lsst/pex/exceptions/Exception.h"
#include "math/Math.h"
#include "math/SurfaceFitting.h"
#include "ndarray/eigen.h"
#include "Spectra.h"
#include "utils/Utils.h"

#define __DEBUGDIR__ ""

namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;

using namespace std;
namespace pfs { namespace drp { namespace stella {

  template <typename T>
  ndarray::Array<T, 1, 1> toArray(std::vector<T> vector) {
    return ndarray::external(vector.data(), ndarray::makeVector(vector.size()), ndarray::makeVector(1));
  }

  template<typename T>
  class PSF {
    public:
      typedef ndarray::Array<T, 1, 1> Vector;
      typedef ndarray::Array<unsigned long, 1, 1> VectorInt;
      typedef ndarray::Array<T, 2, 1> Image;
      typedef math::ThinPlateSpline<T, T> Spline;
      typedef math::ThinPlateSplineChiSquare<T, T> SplineChiSquare;
      typedef struct ExtractPSFResult {
          ExtractPSFResult(std::size_t length);
          Vector xRelativeToCenter;
          Vector yRelativeToCenter;
          Vector zNormalized;
          Vector zTrace;
          Vector weight;
          Vector xTrace;
          Vector yTrace;
          T xCenterPSFCCD;
          T yCenterPSFCCD;
      } ExtractPSFResult;

      explicit PSF(std::size_t iTrace=0, std::size_t iBin=0)
        : _twoDPSFControl(new TwoDPSFControl()),
          _iTrace(iTrace),
          _iBin(iBin),
          _yMin(0),
          _yMax(10),
          _imagePSF_XTrace(0),
          _imagePSF_YTrace(0),
          _imagePSF_ZTrace(0),
          _imagePSF_XRelativeToCenter(0),
          _imagePSF_YRelativeToCenter(0),
          _imagePSF_ZNormalized(0),
          _imagePSF_ZFit(0),
          _imagePSF_Weight(0),
          _xCentersPSFCCD(0),
          _yCentersPSFCCD(0),
          _nPixPerPSF(0),
          _xRangePolynomial(2),
          _isTwoDPSFControlSet(false),
          _isPSFsExtracted(false),
          _thinPlateSpline(),
          _thinPlateSplineChiSquare()
      {};
      
      /**
       * @brief Copy Constructor for a PSF
       * 
       * @param psf: PSF to be copied
       */
      PSF( PSF const& psf )
        : _twoDPSFControl(psf.getTwoDPSFControl()),
          _iTrace(psf.getITrace()),
          _iBin(psf.getIBin()),
          _yMin(psf.getYLow()),
          _yMax(psf.getYHigh()),
          _imagePSF_XTrace(psf.getImagePSF_XTrace()),
          _imagePSF_YTrace(psf.getImagePSF_YTrace()),
          _imagePSF_ZTrace(psf.getImagePSF_ZTrace()),
          _imagePSF_XRelativeToCenter(psf.getImagePSF_XRelativeToCenter()),
          _imagePSF_YRelativeToCenter(psf.getImagePSF_YRelativeToCenter()),
          _imagePSF_ZNormalized(psf.getImagePSF_ZNormalized()),
          _imagePSF_ZFit(psf.getImagePSF_ZFit()),
          _imagePSF_Weight(psf.getImagePSF_Weight()),
          _xCentersPSFCCD(psf.getXCentersPSFCCD()),
          _yCentersPSFCCD(psf.getYCentersPSFCCD()),
          _nPixPerPSF(psf.getNPixPerPSF()),
          _isTwoDPSFControlSet(psf.isTwoDPSFControlSet()),
          _isPSFsExtracted(psf.isPSFsExtracted()),
          _thinPlateSpline(),
          _thinPlateSplineChiSquare()
      {
        #ifdef __DEBUG_PSF__
          cout << "PSF::Copy Constructor started" << endl;
        #endif
        PTR(TwoDPSFControl) ptr(new TwoDPSFControl(*(psf.getTwoDPSFControl())));
        _twoDPSFControl.reset();
        _twoDPSFControl = ptr;
        #ifdef __DEBUG_PSF__
          cout << "PSF::Copy Constructor finished" << endl;
        #endif
      }
      
      /**
       *  @brief Constructor for a PSF
       *
       *  @param[in] yLow               Lower y limit of Bin for which the PSF shall be computed
       *  @param[in] yHigh              Upper y limit of Bin for which the PSF shall be computed
       *  @param[in] twoDPSFControl     Structure containing the parameters for the computation of the PSF
       *  @param[in] iTrace             Trace number for which the PSF shall be computed (for debugging purposes only)
       *  @param[in] iBin               Bin number for which the PSF shall be computed (for debugging purposes only)
       */
      explicit PSF(const std::size_t yLow,
                   const std::size_t yHigh,
                   const PTR(TwoDPSFControl) &twoDPSFControl,
                   std::size_t iTrace = 0,
                   std::size_t iBin = 0)
          : _twoDPSFControl(twoDPSFControl),
            _iTrace(iTrace),
            _iBin(iBin),
            _yMin(yLow),
            _yMax(yHigh),
            _imagePSF_XTrace(0),
            _imagePSF_YTrace(0),
            _imagePSF_ZTrace(0),
            _imagePSF_XRelativeToCenter(0),
            _imagePSF_YRelativeToCenter(0),
            _imagePSF_ZNormalized(0),
            _imagePSF_ZFit(0),
            _imagePSF_Weight(0),
            _xCentersPSFCCD(0),
            _yCentersPSFCCD(0),
            _nPixPerPSF(0),
            _xRangePolynomial(2),
            _isTwoDPSFControlSet(true),
            _isPSFsExtracted(false),
            _thinPlateSpline(),
            _thinPlateSplineChiSquare()
      {};
      
      virtual ~PSF() {};

      std::size_t getIBin() const { return _iBin; }
      std::size_t getITrace() const { return _iTrace; }
      std::size_t getYLow() const { return _yMin; }
      std::size_t getYHigh() const { return _yMax; }
      Vector getImagePSF_XTrace() { return _imagePSF_XTrace; }
      Vector getImagePSF_YTrace() { return _imagePSF_YTrace; }
      Vector getImagePSF_ZTrace() { return _imagePSF_ZTrace; }
      const Vector getImagePSF_XTrace() const { return _imagePSF_XTrace; }
      const Vector getImagePSF_YTrace() const { return _imagePSF_YTrace; }
      const Vector getImagePSF_ZTrace() const { return _imagePSF_ZTrace; }
      Vector getImagePSF_XRelativeToCenter() { return _imagePSF_XRelativeToCenter; }
      Vector getImagePSF_YRelativeToCenter() { return _imagePSF_YRelativeToCenter; }
      Vector getImagePSF_ZNormalized() { return _imagePSF_ZNormalized; }
      Vector getImagePSF_ZFit() { return _imagePSF_ZFit; }
      const Vector getImagePSF_XRelativeToCenter() const { return _imagePSF_XRelativeToCenter; }
      const Vector getImagePSF_YRelativeToCenter() const { return _imagePSF_YRelativeToCenter; }
      const Vector getImagePSF_ZNormalized() const { return _imagePSF_ZNormalized; }
      const Vector getImagePSF_ZFit() const { return _imagePSF_ZFit; }
      Vector getImagePSF_Weight() { return _imagePSF_Weight; }
      Vector getXCentersPSFCCD() { return _xCentersPSFCCD; }
      Vector getYCentersPSFCCD() { return _yCentersPSFCCD; }
      VectorInt getNPixPerPSF() { return _nPixPerPSF; }
      const Vector getImagePSF_Weight() const { return _imagePSF_Weight; }
      const Vector getXCentersPSFCCD() const { return _xCentersPSFCCD; }
      const Vector getYCentersPSFCCD() const { return _yCentersPSFCCD; }
      const VectorInt getNPixPerPSF() const { return _nPixPerPSF; }
      const Vector getXRangePolynomial() const { return _xRangePolynomial; }

      void setImagePSF_ZTrace(Vector const& zTrace);
      void setImagePSF_ZNormalized(Vector const& zNormalized);
      void setImagePSF_ZFit(Vector const& zFit);
      void setXCentersPSFCCD(Vector const& xCentersPSFCCD_In);
      void setYCentersPSFCCD(Vector const& yCentersPSFCCD_In);

      bool isTwoDPSFControlSet() const { return _isTwoDPSFControlSet; }
      bool isPSFsExtracted() const { return _isPSFsExtracted; }
      
      /// Return _2dPSFControl
      PTR( TwoDPSFControl ) getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      void setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl);

      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
      void extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	                   Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In);
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
      void extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	                   Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In,
                       Image const& collapsedPSF);
  
      /**
       * @brief Extract the PSF from the given center positions in CCD coordinates
       * 
       * @param fiberTrace_In: fiberTrace from which this PSF is to be extracted
       * @param centerPositionXCCD_In: PSF center position in x
       * @param centerPositionYCCD_In: PSF center position in y
       */
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      ExtractPSFResult extractPSFFromCenterPosition( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
                                                     T const centerPositionXCCD_In,
                                                     T const centerPositionYCCD_In);
      
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      void extractPSFFromCenterPositions(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
                                         Vector const& centerPositionsX_In,
                                         Vector const& centerPositionsY_In);
      
      /*
       * @brief Use center positions given in this->x/yCentersPSFCCD
       */
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      void extractPSFFromCenterPositions(FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In);
      
      /*
       * @brief fit regularized thin-plate spline to PSF and reconstruct zNormalized from that fit
       */
      Vector reconstructFromThinPlateSplineFit( double const regularization = 0. ) const;
      
      /*
       * @brief Reconstruct zNormalized from given fitted PSF on a x-y grid
       *        Create an exact (regularization == 0.) thin-plate spline from zFit and reconstruct zNormalized from that fit
       * 
       * @param zFit_In: 2D Array of shape ( yGridRelativeToCenterFit_In.shape, xGridRelativeToCenterFit_In.shape )
       */
      Vector reconstructPSFFromFit(Vector const& xGridRelativeToCenterFit_In,
                                   Vector const& yGridRelativeToCenterFit_In,
                                   Image const& zFit_In,
                                   double regularization = 0.) const;
      Vector reconstructPSFFromFit(Vector const& xGridRelativeToCenterFit_In,
                                   Vector const& yGridRelativeToCenterFit_In,
                                   Image const& zFit_In,
                                   Image const& weights_In) const;
      
      /**
       * @brief Fit a fitted PSF (e.g. by thin-plate spline) to minimize the Chi square between the fit and ZTrace
       */
      double fitFittedPSFToZTrace(Vector const& zFit);
      double fitFittedPSFToZTrace(Vector const& zFit,
                                  Vector const& measureErrors);
            
      Spline getThinPlateSpline() const{
          return Spline(_thinPlateSpline);
      }
      void setThinPlateSpline(Spline const& tps){
          _thinPlateSpline = tps;
      }
      SplineChiSquare getThinPlateSplineChiSquare() const{
          return SplineChiSquare(_thinPlateSplineChiSquare);
      }
      void setThinPlateSplineChiSquare(SplineChiSquare const& tps ){
          _thinPlateSplineChiSquare = tps;
      }
      
  protected:

    private:
      PTR(TwoDPSFControl) _twoDPSFControl;
      const std::size_t _iTrace;
      const std::size_t _iBin;
      const std::size_t _yMin;/// first index of swath in fiberTrace
      const std::size_t _yMax;/// last index of swath in fiberTrace
      Vector _imagePSF_XTrace;
      Vector _imagePSF_YTrace;
      Vector _imagePSF_ZTrace;
      Vector _imagePSF_XRelativeToCenter;
      Vector _imagePSF_YRelativeToCenter;
      Vector _imagePSF_ZNormalized;
      Vector _imagePSF_ZFit;
      Vector _imagePSF_Weight;
      Vector _xCentersPSFCCD;
      Vector _yCentersPSFCCD;
      ndarray::Array<unsigned long, 1, 1> _nPixPerPSF;
      Vector _xRangePolynomial;
      bool _isTwoDPSFControlSet;
      bool _isPSFsExtracted;
      Spline _thinPlateSpline;
      SplineChiSquare _thinPlateSplineChiSquare;
      
  };
  
  
/************************************************************************************************************/
/**
 * \brief Describe a set of 2D PSFs
 *
 */
template<typename T>//, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel>//, typename WavelengthT = afwImage::VariancePixel>
class PSFSet {
  public:
    /// Class Constructors and Destructor
      
    /// Creates a new PSFSet object of size 0
    explicit PSFSet(unsigned int nPSFs=0)
        : _psfs(new std::vector<PTR(PSF<T>)>(nPSFs))
        {}
        
    /// Copy constructor
    /// If psfSet is not empty, the object shares ownership of psfSet's PSF vector and increases the use count.
    /// If psfSet is empty, an empty object is constructed (as if default-constructed).
    explicit PSFSet(PSFSet<T> & psfSet)
        : _psfs(psfSet.getPSFs())
        {}
    
    /// Construct an object with a copy of psfVector
    explicit PSFSet(std::vector<PTR(PSF<T>)> & psfVector)
        : _psfs(new std::vector<PTR(PSF<T>)>(psfVector))
        {}
        
    virtual ~PSFSet() {}

    /// Return the number of PSFs
    std::size_t size() const { return _psfs->size(); }

    /// Return the PSF at the ith position
    PTR(PSF<T>) &getPSF(const std::size_t i);

    const PTR(const PSF<T>) getPSF(const std::size_t i) const;

    /**
     * @brief Set the ith PSF
     * @param[in] i     :: which PSF to set
     * @param[in] psf   :: the PSF to be set at the ith position
     * */
    void setPSF(const std::size_t i,
                const PTR(PSF<T>) & psf );

    /// add one PSF to the set
    void addPSF(const PTR(PSF<T>) & psf);

    PTR(std::vector<PTR(PSF<T>)>) getPSFs() const { return _psfs; }
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    void erase(const std::size_t iStart, const std::size_t iEnd=0);

    private:
    PTR(std::vector<PTR(PSF<T>)>) _psfs; // shared pointer to vector of shared pointers to PSFs
};

namespace math{
  template< typename PsfT = double, typename ImageT = afwImage::VariancePixel, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  PTR( PSFSet< PsfT > ) calculate2dPSFPerBin( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                              Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                              TwoDPSFControl const& twoDPSFControl );

  template< typename PsfT = double, typename ImageT = afwImage::VariancePixel, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  PTR( PSFSet< PsfT > ) calculate2dPSFPerBin( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                              Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                              TwoDPSFControl const& twoDPSFControl,
                                              ndarray::Array< PsfT, 2, 1 > const& collapsedPSF);
  
  template< typename PsfT = double, typename ImageT = afwImage::VariancePixel, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  std::vector< PTR( PSFSet< PsfT > ) > calculate2dPSFPerBin( FiberTraceSet< ImageT, MaskT, VarianceT > const& fiberTraceSet,
                                                             SpectrumSet< ImageT, MaskT, VarianceT, WavelengthT > const& spectrumSet,
                                                             TwoDPSFControl const& twoDPSFControl );
  
  /*
   * @brief: fit PSF and interpolate to new coordinates using (regularized) thin-plate splines, reconstruct psf._imagePSF_ZNormalized and write to psf._imagePSF_ZFit
   * 
   * @param psf : PSF to interpolate
   * @param xPositions : x positions of new coordinates relative to center of PSF [x_0, x_1, ... , x_n-2, x_n-1]
   * @param yPositions : y positions of new coordinates relative to center of PSF [y_0, y_1, ... , y_m-2, y_m-1]
   * @param isXYPositionsGridPoints : if yes then output array will have shape [m, n], otherwise m == n and shape of output array will be [n, 1]
   * @param regularization : regularization ( >= 0.) for fit. If equal to 0. the fit will be forced through the original data points
   * @param shapeParameter: additional parameter for RBF, e.g. to solve r^2 * ln( r / shapeParameter) Default = 1
   * @param mode : mode == 0: fit psf._imagePSF_ZNormalized, mode == 1: fit psf._imagePSF_ZTrace
   */
  template< typename PsfT = double, typename CoordsT = double >
  ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSpline( PSF< PsfT > & psf,
                                                              ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                              ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                              bool const isXYPositionsGridPoints,
                                                              double const regularization = 0.,
                                                              PsfT const shapeParameter = 1.,
                                                              unsigned int const mode = 0 );
  
  /*
   * @brief: fit PSF and interpolate to new coordinates using weighted thin-plate splines, reconstruct psf._imagePSF_ZNormalized and write to psf._imagePSF_ZFit
   * 
   * @param psf : PSF to interpolate
   * @param weights : weights ( >= 0.) per input data point for fit. If equal to 0. the fit will be forced through the original data points
   * @param xPositions : x positions of new coordinates relative to center of PSF [x_0, x_1, ... , x_n-2, x_n-1]
   * @param yPositions : y positions of new coordinates relative to center of PSF [y_0, y_1, ... , y_m-2, y_m-1]
   * @param isXYPositionsGridPoints : if yes then output array will have shape [m, n], otherwise m == n and shape of output array will be [n, 1]
   * @param shapeParameter: to solve r^2 * ln( r / radiusNormliazationFactor) Default = 1
   * @param mode : mode == 0: fit psf._imagePSF_ZNormalized, mode == 1: fit psf._imagePSF_ZTrace
   */
  template< typename PsfT = double, typename WeightT = float, typename CoordsT = double>
  ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSpline( PSF< PsfT > & psf,
                                                              ndarray::Array< WeightT, 1, 1 > const& weights,
                                                              ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                              ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                              bool const isXYPositionsGridPoints,
                                                              PsfT const shapeParameter = 1.,
                                                              unsigned int const mode = 0 );
  
  /*
   * @brief: fit PSF and interpolate to new coordinates using thin-plate splines with Chi-square minimization, 
   *         reconstruct psf._imagePSF_ZNormalized and write to psf._imagePSF_ZFit. x/yPositions are the knots
   * 
   * @param psf : PSF to interpolate
   * @param xPositions : x positions of new coordinate grid relative to center of PSF [x_0, x_1, ... , x_n-2, x_n-1]
   * @param yPositions : y positions of new coordinate grid relative to center of PSF [y_0, y_1, ... , y_m-2, y_m-1]
   * @param regularization : regularization parameter for thin-plate spline fitting >= 0 (0 = no regularization)
   * @param shapeParameter: to solve r^2 * ln( r / radiusNormliazationFactor) Default = 1
   * @param mode : mode == 0: fit psf._imagePSF_ZNormalized, mode == 1: fit psf._imagePSF_ZTrace
   */
  template< typename PsfT = double, typename CoordsT = double >
  ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< PsfT > & psf,
                                                                       ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                       ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                       bool const isXYPositionsGridPoints,
                                                                       PsfT const regularization = 0.,
                                                                       PsfT const shapeParameter = 1.,
                                                                       unsigned int const mode = 0 );
  
  template< typename PsfT = double, typename CoordsT = double>
  ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline( PSFSet< PsfT > & psfSet,
                                                                 ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                 ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                 bool const isXYPositionsGridPoints,
                                                                 double const regularization = 0. );
  
  //weightArr: [nPoints, nPSFs]
  template< typename PsfT = double, typename WeightT = float, typename CoordsT = double >
  ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline( PSFSet< PsfT > & psfSet,
                                                                 ndarray::Array< WeightT, 2, 1 > const& weightArr,
                                                                 ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                 ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                 bool const isXYPositionsGridPoints);
  
  /*
   * @brief collapse one fitted PSF in one direction
   * @param xVec_In: vector of x positions of grid
   * @param yVec_In: vector of y positions of grid
   * @param zArr_In: array of z values for x-y grid
   * @param direction: 0: collapse in x (get PSF in dispersion direction)
   *                   1: collapse in y (get PSF in spatial direction)
   * @output [i,0]: coordinate value (y for direction == 0; x for direction ==1)
   *         [i,1]: (sub)pixel value for coordinate position [i,0]
   */
    template< typename PsfT = double, typename CoordT = float >
    ndarray::Array< PsfT, 2, 1 > collapseFittedPSF( ndarray::Array< CoordT, 1, 1 > const& xGridVec_In,
                                                    ndarray::Array< CoordT, 1, 1 > const& yGridVec_In,
                                                    ndarray::Array< PsfT, 2, 1 > const& zArr_In,
                                                    int const direction = 0. );
  
  /*
   * @brief collapse one PSF in one direction
   * @param direction: 0: collapse in x (get PSF in dispersion direction)
   *                   1: collapse in y (get PSF in spatial direction)
   */
  template< typename PsfT = double, typename CoordT = double >
  ndarray::Array< PsfT, 2, 1 > collapsePSF( PSF< PsfT > const& psf_In,
                                            ndarray::Array< CoordT, 1, 1 > const& coordinatesX_In,
                                            ndarray::Array< CoordT, 1, 1 > const& coordinatesY_In,
                                            int const direction = 0.,
                                            double const regularization = 0.);
  
  /*
   * @brief convert absolute coordinates from [0,...,N] to coordinates relative to center position in range(centerPos_In - width_In/2., centerPos_In + width_In/2.)
   * @param centerPos_In: center position to convert coords_In relative to
   * @param width_In: all pixels touched by the limits will be in output array 
   */
  template< typename T >
  ndarray::Array< T, 2, 1> calcPositionsRelativeToCenter(T const centerPos_In,
                                                         T const width_In);

  /* @brief compare center positions of emission lines used to construct PSF to input list of x and y positions and return dx, dy, dr
   * @param psf_In Input PSF to compare center positions of emission lines to xPositions_In and yPositions_In
   * @param xPositions_In Input list of center positions of emission lines in x CCD
   * @param yPositions_In Input list of center positions of emission lines in y CCD
   * @param dXMax maximum distance in x in pixels to count 2 emission lines in psf_In and x/yPositions_In as the same emission line
   * @param dYMax maximum distance in y in pixels to count 2 emission lines in psf_In and x/yPositions_In as the same emission line
   * @return one line per emission line used to construct PSF with [dx=xPositions[iList] - xPSF, dy=yPositions[iList] - yPSF, dr=sqrt(pow(dx)+pow(dy))]
   */
  template< typename PsfT = double, typename CoordsT = double >
  ndarray::Array< CoordsT, 2, 1 > compareCenterPositions(PSF< PsfT > & psf_In,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& xPositions_In,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& yPositions_In,
                                                         float dXMax = 1.,
                                                         float dYMax = 1.,
                                                         bool setPsfXY = false);

}
}}}
#include "PSFTemplates.hpp"
#endif
