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
#include "SurfaceFit.h"
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
#define __DEBUGDIR__ ""//~/spectra/pfs/2014-11-02/debug/"// 

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;
//namespace blas = boost::numeric;//::ublas;

using namespace std;
namespace pfs { namespace drp { namespace stella {

  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel, typename WavelengthT=afwImage::VariancePixel>
  class PSF {
    public:
      typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

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
                                                     _pixelsFit(0),
                                                     _isTwoDPSFControlSet(false),
                                                     _isPSFsExtracted(false),
                                                     _surfaceFit()
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
       *  @param[in] trace              Masked image of a FiberTrace for which to compute PSF in one bin specified by yLow_In and yHigh_In
       *  @param[in] xCenters           Vector containing the xCenters of the FiberTrace
       *  @param[in] yLow               Lower y limit of Bin for which the PSF shall be computed
       *  @param[in] yHigh              Upper y limit of Bin for which the PSF shall be computed
       *  @param[in] twoDPSFControl     Structure containing the parameters for the computation of the PSF
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
            _pixelsFit(0),
            _isTwoDPSFControlSet(true),
            _isPSFsExtracted(false),
            _surfaceFit()
      {};
      
      virtual ~PSF() {};

      /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
//      virtual PTR(PSF) clone() const;

      /// Return the dimensions of the images returned by computeImage()
//      geom::Extent2I getDimensions() const { return _dimensions; }

      /// Whether the Psf is persistable; always true.
//      virtual bool isPersistable() const { return true; }
      size_t getIBin() const {return _iBin;}
      size_t getITrace() const {return _iTrace;}
      size_t getYLow() const {return _yMin;}
      size_t getYHigh() const {return _yMax;}
      std::vector<float> getImagePSF_XTrace() {return _imagePSF_XTrace;}
      std::vector<float> getImagePSF_YTrace() {return _imagePSF_YTrace;}
      std::vector<float> getImagePSF_ZTrace() {return _imagePSF_ZTrace;}
      const std::vector<float> getImagePSF_XTrace() const {return _imagePSF_XTrace;}
      const std::vector<float> getImagePSF_YTrace() const {return _imagePSF_YTrace;}
      const std::vector<float> getImagePSF_ZTrace() const {return _imagePSF_ZTrace;}
      std::vector<float> getImagePSF_XRelativeToCenter() {return _imagePSF_XRelativeToCenter;}
      std::vector<float> getImagePSF_YRelativeToCenter() {return _imagePSF_YRelativeToCenter;}
      std::vector<float> getImagePSF_ZNormalized() {return _imagePSF_ZNormalized;}
      const std::vector<float> getImagePSF_XRelativeToCenter() const {return _imagePSF_XRelativeToCenter;}
      const std::vector<float> getImagePSF_YRelativeToCenter() const {return _imagePSF_YRelativeToCenter;}
      const std::vector<float> getImagePSF_ZNormalized() const {return _imagePSF_ZNormalized;}
      std::vector<float> getImagePSF_Weight() {return _imagePSF_Weight;}
      std::vector<float> getPixelsFit() {return _pixelsFit;}
      const std::vector<float> getImagePSF_Weight() const {return _imagePSF_Weight;}
      const std::vector<float> getPixelsFit() const {return _pixelsFit;}
      bool isTwoDPSFControlSet() const {return _isTwoDPSFControlSet;}
      bool isPSFsExtracted() const {return _isPSFsExtracted;}
      SurfaceFit getSurfaceFit() const {return _surfaceFit;}
      
      /// Return _2dPSFControl
      PTR(TwoDPSFControl) getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl);

      /// Return the SurfaceFit
//      PTR(SurfaceFit) getSurfaceFit() const {return boost::make_shared<SurfaceFit>(_surfaceFit);}

      bool extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In);
      bool extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace_In,
	               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum_In,
                       ndarray::Array<ImageT, 2, 1> const& collapsedPSF);
      bool fitPSFKernel();
      bool calculatePSF();
      
      std::vector< ImageT > reconstructFromThinPlateSplineFit(double const regularization = 0.);
      
  protected:

//    virtual std::string getPersistenceName() const;

//    virtual std::string getPythonModule() const;

//    virtual void write(OutputArchiveHandle & handle) const;

    private:
      PTR(TwoDPSFControl) _twoDPSFControl;
      const size_t _iTrace;
      const size_t _iBin;
      const size_t _yMin;
      const size_t _yMax;
      std::vector<float> _imagePSF_XTrace;
      std::vector<float> _imagePSF_YTrace;
      std::vector<float> _imagePSF_ZTrace;
      std::vector<float> _imagePSF_XRelativeToCenter;
      std::vector<float> _imagePSF_YRelativeToCenter;
      std::vector<float> _imagePSF_ZNormalized;
      std::vector<float> _imagePSF_Weight;
      std::vector<float> _pixelsFit;
      bool _isTwoDPSFControlSet;
      bool _isPSFsExtracted;
      SurfaceFit _surfaceFit;
      
  };
  
  
/************************************************************************************************************/
/**
 * \brief Describe a set of 2D PSFs
 *
 */
template<typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
class PSFSet {
  public:
    /// Class Constructors and Destructor
      
    /// Creates a new PSFSet object of size 0
    explicit PSFSet(unsigned int nPSFs=0)
        : _psfs(new std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>(nPSFs))
        {}
        
    /// Copy constructor
    /// If psfSet is not empty, the object shares ownership of psfSet's PSF vector and increases the use count.
    /// If psfSet is empty, an empty object is constructed (as if default-constructed).
    explicit PSFSet(PSFSet<ImageT, MaskT, VarianceT, WavelengthT> & psfSet)
        : _psfs(psfSet.getPSFs())
        {}
    
    /// Construct an object with a copy of psfVector
    explicit PSFSet(std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)> & psfVector)
        : _psfs(new std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>(psfVector))
        {}
        
    virtual ~PSFSet() {}

    /// Return the number of PSFs
    size_t size() const { return _psfs->size(); }

    /// Return the PSF at the ith position
    PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) &getPSF(const size_t i);

    const PTR(const PSF<ImageT, MaskT, VarianceT, WavelengthT>) getPSF(const size_t i) const;

    /// Set the ith PSF
    bool setPSF(const size_t i,     /// which spectrum?
                const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf /// the PSF at the ith position
                      );

    /// add one PSF to the set
    void addPSF(const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf);

    PTR(std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>) getPSFs() const { return _psfs; }
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const size_t iStart, const size_t iEnd=0);

    private:
    PTR(std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>) _psfs; // shared pointer to vector of shared pointers to PSFs
};

namespace math{
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  PTR( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > ) calculate2dPSFPerBin( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                                                               Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                                                               TwoDPSFControl const& twoDPSFControl );

  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  PTR( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > ) calculate2dPSFPerBin( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                                                               Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                                                               TwoDPSFControl const& twoDPSFControl,
                                                                               ndarray::Array< ImageT, 2, 1 > const& collapsedPSF);
  
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  std::vector< PTR( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > ) > calculate2dPSFPerBin( FiberTraceSet< ImageT, MaskT, VarianceT > const& fiberTraceSet,
                                                                                              SpectrumSet< ImageT, MaskT, VarianceT, WavelengthT > const& spectrumSet,
                                                                                              TwoDPSFControl const& twoDPSFControl );
  
  /*
   * @brief: fit PSF and interpolate to new grid using thin-plate splines. The output PSF will have n x m data points
   * 
   * @param xPositions: x positions of new grid relative to center of PSF [x_0, x_1, ... , x_n-2, x_n-1]
   * @param xPositions: y positions of new grid relative to center of PSF [y_0, y_1, ... , y_m-2, y_m-1]
   */
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  ndarray::Array< ImageT, 2, 1 > interpolatePSFThinPlateSpline( PSF< ImageT, MaskT, VarianceT, WavelengthT > const& psf,
                                                                ndarray::Array< float, 1, 1 > const& xPositions,
                                                                ndarray::Array< float, 1, 1 > const& yPositions,
                                                                bool const isXYPositionsGridPoints,
                                                                double const regularization = 0. );
  
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  ndarray::Array< ImageT, 2, 1 > interpolatePSFThinPlateSpline( PSF< ImageT, MaskT, VarianceT, WavelengthT > const& psf,
                                                                ndarray::Array< float, 1, 1 > const& weights,
                                                                ndarray::Array< float, 1, 1 > const& xPositions,
                                                                ndarray::Array< float, 1, 1 > const& yPositions,
                                                                bool const isXYPositionsGridPoints,
                                                                double const regularization = 0. );
  
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  ndarray::Array< ImageT, 3, 1 > interpolatePSFSetThinPlateSpline( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > const& psfSet,
                                                                   ndarray::Array< float, 1, 1 > const& xPositions,
                                                                   ndarray::Array< float, 1, 1 > const& yPositions,
                                                                   bool const isXYPositionsGridPoints,
                                                                   double const regularization = 0. );
  
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  ndarray::Array< ImageT, 3, 1 > interpolatePSFSetThinPlateSpline( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > const& psfSet,
                                                                   ndarray::Array< float, 2, 1 > const& weightArr,
                                                                   ndarray::Array< float, 1, 1 > const& xPositions,
                                                                   ndarray::Array< float, 1, 1 > const& yPositions,
                                                                   bool const isXYPositionsGridPoints,
                                                                   double const regularization = 0. );
  
  /*
   * @brief collapse one fitted PSF in one direction
   * @param xVec_In: vector of x positions of grid
   * @param yVec_In: vector of y positions of grid
   * @param zArr_In: array of z values for x-y grid
   * @param direction: 0: collapse in x (get PSF in dispersion direction)
   *                   1: collapse in y (get PSF in spatial direction)
   */
    template< typename ImageT, typename CoordT = float >
    ndarray::Array< ImageT, 2, 1 > collapseFittedPSF( ndarray::Array< CoordT, 1, 1 > const& xGridVec_In,
                                                      ndarray::Array< CoordT, 1, 1 > const& yGridVec_In,
                                                      ndarray::Array< ImageT, 2, 1 > const& zArr_In,
                                                      int const direction = 0. );
  
  /*
   * @brief collapse one fitted PSF in one direction
   * @param direction: 0: collapse in x (get PSF in dispersion direction)
   *                   1: collapse in y (get PSF in spatial direction)
   */
  template< typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel >
  ndarray::Array< float, 2, 1 > collapsePSF( PSF< ImageT, MaskT, VarianceT, WavelengthT > const& psf_In,
                                             ndarray::Array< float, 1, 1 > const& coordinatesX_In,
                                             ndarray::Array< float, 1, 1 > const& coordinatesY_In,
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
  
}
}}}
#endif
