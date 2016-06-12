///TODO: calculate2dPSF: remove outliers in PSF

#ifndef __PFS_DRP_STELLA_PSF_H__
#define __PFS_DRP_STELLA_PSF_H__

#include <vector>
#include <iostream>
#include <cassert>
#include "lsst/base.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions/Exception.h"
#include "Controls.h"
#include "utils/Utils.h"
#include "math/Math.h"
#include "math/LinearAlgebra3D.h"
#include "math/SurfaceFitting.h"
//#include "SurfaceFit.h"
#include "cmpfit-1.2/MPFitting_ndarray.h"
#include "FiberTraces.h"
#include "Spectra.h"
#include "boost/make_shared.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "ndarray/eigen.h"

//#include "lsst/afw/table/io/InputArchive.h"
//#include "lsst/afw/table/io/OutputArchive.h"
//#include "lsst/afw/table/io/CatalogVector.h"

//#define __DEBUG_CALC2DPSF__
//#define __DEBUG_CPRTC__
//#define __DEBUG_CALC_TPS__
//#define __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
//#define __DEBUG_COMPARECENTERPOSITIONS__
//#define __DEBUG_PSF__
#define __DEBUGDIR__ ""//~/spectra/pfs/2014-11-02/debug/"// 

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;
//namespace blas = boost::numeric;//::ublas;

using namespace std;
namespace pfs { namespace drp { namespace stella {
    
  template< typename T >
  struct ExtractPSFResult{
    std::vector<T> xRelativeToCenter;
    std::vector<T> yRelativeToCenter;
    std::vector<T> zNormalized;
    std::vector<T> zTrace;
    std::vector<T> weight;
    std::vector<T> xTrace;
    std::vector<T> yTrace;
    T xCenterPSFCCD;
    T yCenterPSFCCD;
  };

  template<typename T>//, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>//, typename WavelengthT=afwImage::VariancePixel>
  class PSF {
    public:
//      typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

      explicit PSF(size_t iTrace=0, size_t iBin=0) : _twoDPSFControl(new TwoDPSFControl()),
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
       * @brief Copy Constructor (shallow and deep) for a PSF
       * 
       * @param psf: PSF to be copied
       * @param deep: type of copy. NOTE that the only shallow copy of psf is psf._twoDPSFControl
       */
      PSF(PSF &psf, const bool deep = false);
      
      /**
       *  @brief Constructor for a PSF
       *
       *  @param[in] yLow               Lower y limit of Bin for which the PSF shall be computed
       *  @param[in] yHigh              Upper y limit of Bin for which the PSF shall be computed
       *  @param[in] twoDPSFControl     Structure containing the parameters for the computation of the PSF
       *  @param[in] iTrace             Trace number for which the PSF shall be computed (for debugging purposes only)
       *  @param[in] iBin               Bin number for which the PSF shall be computed (for debugging purposes only)
       */
      explicit PSF(const size_t yLow,
                   const size_t yHigh,
                   const PTR(TwoDPSFControl) &twoDPSFControl,
                   size_t iTrace = 0,
                   size_t iBin = 0)
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
      
      /**
       *  @brief Constructor for a PSF
       *
       *  @param[in] yLow               Lower y limit of Bin for which the PSF shall be computed
       *  @param[in] yHigh              Upper y limit of Bin for which the PSF shall be computed
       *  @param[in] twoDPSFControl     Structure containing the parameters for the computation of the PSF
       *  @param[in] iTrace             Trace number for which the PSF shall be computed (for debugging purposes only)
       *  @param[in] iBin               Bin number for which the PSF shall be computed (for debugging purposes only)
       */
      explicit PSF( const long yLow,
                    const long yHigh,
                    const PTR(TwoDPSFControl) &twoDPSFControl,
                    long iTrace = 0,
                    long iBin = 0 )
          : _twoDPSFControl( twoDPSFControl ),
            _iTrace( size_t( iTrace ) ),
            _iBin( size_t( iBin ) ),
            _yMin( size_t( yLow ) ),
            _yMax( size_t( yHigh ) ),
            _imagePSF_XTrace(0),
            _imagePSF_YTrace(0),
            _imagePSF_ZTrace(0),
            _imagePSF_XRelativeToCenter(0),
            _imagePSF_YRelativeToCenter(0),
            _imagePSF_ZNormalized(0),
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

      /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
//      virtual PTR(PSF) clone() const;

      /// Return the dimensions of the images returned by computeImage()
//      geom::Extent2I getDimensions() const { return _dimensions; }

      /// Whether the Psf is persistable; always true.
//      virtual bool isPersistable() const { return true; }
      size_t getIBin() const { return _iBin; }
      size_t getITrace() const { return _iTrace; }
      size_t getYLow() const { return _yMin; }
      size_t getYHigh() const { return _yMax; }
      std::vector< T > getImagePSF_XTrace() { return _imagePSF_XTrace; }
      std::vector< T > getImagePSF_YTrace() { return _imagePSF_YTrace; }
      std::vector< T > getImagePSF_ZTrace() { return _imagePSF_ZTrace; }
      const std::vector< T > getImagePSF_XTrace() const { return _imagePSF_XTrace; }
      const std::vector< T > getImagePSF_YTrace() const { return _imagePSF_YTrace; }
      const std::vector< T > getImagePSF_ZTrace() const { return _imagePSF_ZTrace; }
      std::vector< T > getImagePSF_XRelativeToCenter() { return _imagePSF_XRelativeToCenter; }
      std::vector< T > getImagePSF_YRelativeToCenter() { return _imagePSF_YRelativeToCenter; }
      std::vector< T > getImagePSF_ZNormalized() { return _imagePSF_ZNormalized; }
      std::vector< T > getImagePSF_ZFit() { return _imagePSF_ZFit; }
      const std::vector< T > getImagePSF_XRelativeToCenter() const { return _imagePSF_XRelativeToCenter; }
      const std::vector< T > getImagePSF_YRelativeToCenter() const { return _imagePSF_YRelativeToCenter; }
      const std::vector< T > getImagePSF_ZNormalized() const { return _imagePSF_ZNormalized; }
      const std::vector< T > getImagePSF_ZFit() const { return _imagePSF_ZFit; }
      std::vector< T > getImagePSF_Weight() { return _imagePSF_Weight; }
      std::vector< T > getXCentersPSFCCD() { return _xCentersPSFCCD; }
      std::vector< T > getYCentersPSFCCD() { return _yCentersPSFCCD; }
      std::vector< unsigned long > getNPixPerPSF() { return _nPixPerPSF; }
      const std::vector< T > getImagePSF_Weight() const { return _imagePSF_Weight; }
      const std::vector< T > getXCentersPSFCCD() const { return _xCentersPSFCCD; }
      const std::vector< T > getYCentersPSFCCD() const { return _yCentersPSFCCD; }
      const std::vector< unsigned long > getNPixPerPSF() const { return _nPixPerPSF; }
      const std::vector< T > getXRangePolynomial() const { return _xRangePolynomial; }
      
      bool setImagePSF_ZTrace( ndarray::Array< T, 1, 1 > const& zTrace);
      bool setImagePSF_ZTrace( std::vector< T > const& zTrace);
      bool setImagePSF_ZNormalized( ndarray::Array< T, 1, 1 > const& zNormalized);
      bool setImagePSF_ZNormalized( std::vector< T > const& zNormalized);
      bool setImagePSF_ZFit( ndarray::Array< T, 1, 1 > const& zFit );
      bool setXCentersPSFCCD( std::vector< T > const& xCentersPSFCCD_In );
      bool setYCentersPSFCCD( std::vector< T > const& yCentersPSFCCD_In );

      bool isTwoDPSFControlSet() const { return _isTwoDPSFControlSet; }
      bool isPSFsExtracted() const { return _isPSFsExtracted; }
      
      /// Return _2dPSFControl
      PTR( TwoDPSFControl ) getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl);

      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
      bool extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In);
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
      bool extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In,
                       ndarray::Array<T, 2, 1> const& collapsedPSF);
  
      /**
       * @brief Extract the PSF from the given center positions in CCD coordinates
       * 
       * @param fiberTrace_In: fiberTrace from which this PSF is to be extracted
       * @param centerPositionX_In: PSF center position in x
       * @param centerPositionY_In: PSF center position in y
       */
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      ExtractPSFResult<T> extractPSFFromCenterPosition( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
                                                        T const centerPositionX_In,
                                                        T const centerPositionY_In);
      
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      bool extractPSFFromCenterPositions( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
                                          ndarray::Array< T, 1, 1 > const& centerPositionsX_In,
                                          ndarray::Array< T, 1, 1 > const& centerPositionsY_In);
      
      /*
       * @brief Use center positions given in this->x/yCentersPSFCCD
       */
      template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel >
      bool extractPSFFromCenterPositions(FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In);
      //bool fitPSFKernel();
      //bool calculatePSF();
      
      /*
       * @brief fit regularized thin-plate spline to PSF and reconstruct zNormalized from that fit
       */
      std::vector< T > reconstructFromThinPlateSplineFit( double const regularization = 0. ) const;
      
      /*
       * @brief Reconstruct zNormalized from given fitted PSF on a x-y grid
       *        Create an exact (regularization == 0.) thin-plate spline from zFit and reconstruct zNormalized from that fit
       * 
       * @param zFit_In: 2D Array of shape ( yGridRelativeToCenterFit_In.shape, xGridRelativeToCenterFit_In.shape )
       */
      std::vector< T > reconstructPSFFromFit( ndarray::Array< T, 1, 1 > const& xGridRelativeToCenterFit_In,
                                              ndarray::Array< T, 1, 1 > const& yGridRelativeToCenterFit_In,
                                              ndarray::Array< T, 2, 1 > const& zFit_In,
                                              double regularization = 0.) const;
      std::vector< T > reconstructPSFFromFit( ndarray::Array< T, 1, 1 > const& xGridRelativeToCenterFit_In,
                                              ndarray::Array< T, 1, 1 > const& yGridRelativeToCenterFit_In,
                                              ndarray::Array< T, 2, 1 > const& zFit_In,
                                              ndarray::Array< T, 2, 1 > const& weights_In) const;
      
      /**
       * @brief Fit a fitted PSF (e.g. by thin-plate spline) to minimize the Chi square between the fit and ZTrace
       */
      double fitFittedPSFToZTrace( std::vector< T > const& zFit_In );
      double fitFittedPSFToZTrace( std::vector< T > const& zFit_In,
                                   std::vector< T > const& measureErrors_In );
      double fitFittedPSFToZTrace( ndarray::Array< T, 1, 1 > const& zFit_In );
      double fitFittedPSFToZTrace( ndarray::Array< T, 1, 1 > const& zFit_In,
                                   ndarray::Array< T, 1, 1 > const& measureErrors_In );
            
      math::ThinPlateSpline< T, T > getThinPlateSpline() const{
          return math::ThinPlateSpline< T, T >( _thinPlateSpline );
      }
      void setThinPlateSpline( math::ThinPlateSpline< T, T > const& tps ){
          _thinPlateSpline = tps;
          return;
      }
      math::ThinPlateSplineChiSquare< T, T > getThinPlateSplineChiSquare() const{
          return math::ThinPlateSplineChiSquare< T, T >( _thinPlateSplineChiSquare );
      }
      void setThinPlateSplineChiSquare( math::ThinPlateSplineChiSquare< T, T > const& tps ){
          _thinPlateSplineChiSquare = tps;
          return;
      }
      
  protected:

//    virtual std::string getPersistenceName() const;

//    virtual std::string getPythonModule() const;

//    virtual void write(OutputArchiveHandle & handle) const;

    private:
      PTR(TwoDPSFControl) _twoDPSFControl;
      const size_t _iTrace;
      const size_t _iBin;
      const size_t _yMin;/// first index of swath in fiberTrace
      const size_t _yMax;/// last index of swath in fiberTrace
      std::vector<T> _imagePSF_XTrace;
      std::vector<T> _imagePSF_YTrace;
      std::vector<T> _imagePSF_ZTrace;
      std::vector<T> _imagePSF_XRelativeToCenter;
      std::vector<T> _imagePSF_YRelativeToCenter;
      std::vector<T> _imagePSF_ZNormalized;
      std::vector<T> _imagePSF_ZFit;
      std::vector<T> _imagePSF_Weight;
      std::vector<T> _xCentersPSFCCD;
      std::vector<T> _yCentersPSFCCD;
      std::vector<unsigned long> _nPixPerPSF;
      std::vector<T> _xRangePolynomial;
      bool _isTwoDPSFControlSet;
      bool _isPSFsExtracted;
      math::ThinPlateSpline< T, T > _thinPlateSpline;
      math::ThinPlateSplineChiSquare< T, T > _thinPlateSplineChiSquare;
      
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
    size_t size() const { return _psfs->size(); }

    /// Return the PSF at the ith position
    PTR(PSF<T>) &getPSF(const size_t i);

    const PTR(const PSF<T>) getPSF(const size_t i) const;

    /**
     * @brief Set the ith PSF
     * @param[in] i     :: which PSF to set
     * @param[in] psf   :: the PSF to be set at the ith position
     * */
    bool setPSF(const size_t i,
                const PTR(PSF<T>) & psf );

    /// add one PSF to the set
    void addPSF(const PTR(PSF<T>) & psf);

    PTR(std::vector<PTR(PSF<T>)>) getPSFs() const { return _psfs; }
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const size_t iStart, const size_t iEnd=0);

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
                                                              unsigned short const mode = 0 );
  
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
                                                              unsigned short const mode = 0 );
  
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
                                                                       unsigned short const mode = 0 );
  
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
#endif
