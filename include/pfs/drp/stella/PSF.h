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
#include "blitz.h"
//#include <cassert>
#include "Controls.h"
#include "utils/Utils.h"
#include "math/Math.h"
#include "SurfaceFit.h"
#include "cmpfit-1.2/MyFit.h"
#include "FiberTraces.h"
#include "Spectra.h"

#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"

//#include "lsst/afw/table/io/InputArchive.h"
//#include "lsst/afw/table/io/OutputArchive.h"
//#include "lsst/afw/table/io/CatalogVector.h"

//#define __DEBUG_CALC2DPSF__
#define __DEBUGDIR__ ""//~/spectra/pfs/2014-11-02/debug/"// 

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella {

  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel, typename WavelengthT=afwImage::VariancePixel>
  class PSF {
    public:
      typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

      PSF(unsigned int iTrace=0, unsigned int iBin=0) : _twoDPSFControl(new TwoDPSFControl()),
                                                        _iTrace(iTrace),
                                                        _iBin(iBin),
                                                        _yLow(0),
                                                        _yHigh(0),
                                                        _imagePSF_XTrace(new std::vector<float>(0)),
                                                        _imagePSF_YTrace(new std::vector<float>(0)),
                                                        _imagePSF_ZTrace(new std::vector<float>(0)),
                                                        _imagePSF_XRelativeToCenter(new std::vector<float>(0)),
                                                        _imagePSF_YRelativeToCenter(new std::vector<float>(0)),
                                                        _imagePSF_ZNormalized(new std::vector<float>(0)),
                                                        _imagePSF_Weight(new std::vector<float>(0)),
                                                        _pixelsFit(new std::vector<float>(0)),
                                                        _isTwoDPSFControlSet(false),
                                                        _isPSFsExtracted(false),
                                                        _surfaceFit(new SurfaceFit())
      {};
      
      PSF(const PSF &psf) : _twoDPSFControl(psf.getTwoDPSFControl()),
                            _iTrace(psf.getITrace()),
                            _iBin(psf.getIBin()),
                            _yLow(psf.getYLow()),
                            _yHigh(psf.getYHigh()),
                            _imagePSF_XTrace(psf.getImagePSF_XTrace()),
                            _imagePSF_YTrace(psf.getImagePSF_YTrace()),
                            _imagePSF_ZTrace(psf.getImagePSF_ZTrace()),
                            _imagePSF_XRelativeToCenter(psf.getImagePSF_XRelativeToCenter()),
                            _imagePSF_YRelativeToCenter(psf.getImagePSF_YRelativeToCenter()),
                            _imagePSF_ZNormalized(psf.getImagePSF_ZNormalized()),
                            _imagePSF_Weight(psf.getImagePSF_Weight()),
                            _pixelsFit(psf.getPixelsFit()),
                            _isTwoDPSFControlSet(psf.isTwoDPSFControlSet()),
                            _isPSFsExtracted(psf.isPSFsExtracted()),
                            _surfaceFit(psf.getSurfaceFit()) {};
      
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
      PSF(const unsigned int yLow,
          const unsigned int yHigh,
          const PTR(TwoDPSFControl) &twoDPSFControl,
          unsigned int iTrace = 0,
          unsigned int iBin = 0
      ) : _twoDPSFControl(twoDPSFControl),
          _iTrace(iTrace),
          _iBin(iBin),
          _yLow(yLow),
          _yHigh(yHigh),
          _imagePSF_XTrace(new std::vector<float>(0)),
          _imagePSF_YTrace(new std::vector<float>(0)),
          _imagePSF_ZTrace(new std::vector<float>(0)),
          _imagePSF_XRelativeToCenter(new std::vector<float>(0)),
          _imagePSF_YRelativeToCenter(new std::vector<float>(0)),
          _imagePSF_ZNormalized(new std::vector<float>(0)),
          _imagePSF_Weight(new std::vector<float>(0)),
          _pixelsFit(new std::vector<float>(0)),
          _isTwoDPSFControlSet(true),
          _isPSFsExtracted(false),
          _surfaceFit(new SurfaceFit())
      {};
      
      virtual ~PSF() {};

      /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
//      virtual PTR(PSF) clone() const;

      /// Return the dimensions of the images returned by computeImage()
//      geom::Extent2I getDimensions() const { return _dimensions; }

      /// Whether the Psf is persistable; always true.
//      virtual bool isPersistable() const { return true; }
      unsigned int getIBin() const {return _iBin;}
      unsigned int getITrace() const {return _iTrace;}
      unsigned int getYLow() const {return _yLow;}
      unsigned int getYHigh() const {return _yHigh;}
      PTR(std::vector<float>) getImagePSF_XTrace() const {return _imagePSF_XTrace;}
      PTR(std::vector<float>) getImagePSF_YTrace() const {return _imagePSF_YTrace;}
      PTR(std::vector<float>) getImagePSF_ZTrace() const {return _imagePSF_ZTrace;}
      PTR(std::vector<float>) getImagePSF_XRelativeToCenter() const {return _imagePSF_XRelativeToCenter;}
      PTR(std::vector<float>) getImagePSF_YRelativeToCenter() const {return _imagePSF_YRelativeToCenter;}
      PTR(std::vector<float>) getImagePSF_ZNormalized() const {return _imagePSF_ZNormalized;}
      PTR(std::vector<float>) getImagePSF_Weight() const {return _imagePSF_Weight;}
      PTR(std::vector<float>) getPixelsFit() const {return _pixelsFit;}
      bool isTwoDPSFControlSet() const {return _isTwoDPSFControlSet;}
      bool isPSFsExtracted() const {return _isPSFsExtracted;}
      PTR(SurfaceFit) getSurfaceFit() const {return _surfaceFit;}
      
      /// Return _2dPSFControl
      PTR(TwoDPSFControl) getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl);

      /// Return the SurfaceFit
//      PTR(SurfaceFit) getSurfaceFit() const {return boost::make_shared<SurfaceFit>(_surfaceFit);}

      bool extractPSFs(const FiberTrace<ImageT, MaskT, VarianceT> &fiberTrace_In,
	               const Spectrum<ImageT, MaskT, VarianceT, WavelengthT> &spectrum_In);
      bool fitPSFKernel();
      bool calculatePSF();
  protected:

//    virtual std::string getPersistenceName() const;

//    virtual std::string getPythonModule() const;

//    virtual void write(OutputArchiveHandle & handle) const;

    private:
      const PTR(TwoDPSFControl) _twoDPSFControl;
      const unsigned int _iTrace;
      const unsigned int _iBin;
      const unsigned int _yLow;
      const unsigned int _yHigh;
      PTR(std::vector<float>) _imagePSF_XTrace;
      PTR(std::vector<float>) _imagePSF_YTrace;
      PTR(std::vector<float>) _imagePSF_ZTrace;
      PTR(std::vector<float>) _imagePSF_XRelativeToCenter;
      PTR(std::vector<float>) _imagePSF_YRelativeToCenter;
      PTR(std::vector<float>) _imagePSF_ZNormalized;
      PTR(std::vector<float>) _imagePSF_Weight;
      PTR(std::vector<float>) _pixelsFit;
      bool _isTwoDPSFControlSet;
      bool _isPSFsExtracted;
      PTR(SurfaceFit) _surfaceFit;
      
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
    int size() const { return _psfs->size(); }

    /// Return the PSF at the ith position
    PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) &getPSF(const unsigned int i ///< desired position
                                                           );

    PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) const& getPSF(const unsigned int i ///< desired position
                                                                 ) const;

    /// Set the ith PSF
    bool setPSF(const unsigned int i,     /// which spectrum?
                const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf /// the PSF at the ith position
                      );

    /// add one PSF to the set
    void addPSF(const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf /// the PSF to add
               );

    PTR(std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>) getPSFs() const { return _psfs; }
    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const unsigned int iStart, const unsigned int iEnd=0);

    private:
    PTR(std::vector<PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)>) _psfs; // shared pointer to vector of shared pointers to PSFs
};

namespace math{
  template<typename ImageT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
  PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) calculate2dPSFPerBin(const FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace,
                                                                          const Spectrum<ImageT, MaskT, VarianceT, WavelengthT> & spectrum,
                                                                          const PTR(TwoDPSFControl) & twoDPSFControl);
}
}}}
#endif
