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
//#include <fitsio.h>
//#include <fitsio2.h>
//#include "cmpfit-1.2/MyFit.h"
//#include "spline.h"
//#include <cassert>
#include "Controls.h"
#include "utils/Utils.h"
#include "math/Math.h"
#include "SurfaceFit.h"
#include "cmpfit-1.2/MyFit.h"
//#include "FiberTraces.h"
//#include "lsst/afw/detection/Psf.h"

#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"

//#include "lsst/afw/table/io/InputArchive.h"
//#include "lsst/afw/table/io/OutputArchive.h"
//#include "lsst/afw/table/io/CatalogVector.h"

#define __DEBUG_CALC2DPSF__
#define DEBUGDIR "/home/azuri/spectra/pfs/2014-11-02/debug/"// 

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella {

  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
  class PSF {
    public:
      typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

      PSF(unsigned int iTrace=0, unsigned int iBin=0) : _trace(),
                                                        _twoDPSFControl(),
                                                        _iTrace(iTrace),
                                                        _iBin(iBin),
                                                        _xLow(0),
                                                        _xHigh(0),
                                                        _yLow(0),
                                                        _yHigh(0),
                                                        _xCenters(1),
                                                        _spectrum(1),
                                                        _spectrumVariance(1),
                                                        _imagePSF_XTrace(1),
                                                        _imagePSF_YTrace(1),
                                                        _imagePSF_ZTrace(1),
                                                        _imagePSF_XRelativeToCenter(1),
                                                        _imagePSF_YRelativeToCenter(1),
                                                        _imagePSF_ZNormalized(1),
                                                        _imagePSF_Weight(1),
                                                        _pixelsFit(1),
                                                        _isTwoDPSFControlSet(false),
                                                        _isPSFsExtracted(false),
                                                        _surfaceFit()
      {};
      
      PSF(const PSF &psf) : _trace(*(psf.getTrace()), true),
                            _twoDPSFControl(psf.getTwoDPSFControl()),
                            _iTrace(psf.getITrace()),
                            _iBin(psf.getIBin()),
                            _xLow(psf.getXLow()),
                            _xHigh(psf.getXHigh()),
                            _yLow(psf.getYLow()),
                            _yHigh(psf.getYHigh()),
                            _xCenters(psf.getXCenters()),
                            _spectrum(psf.getSpectrum()),
                            _spectrumVariance(psf.getSpectrumVariance()),
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
                            _surfaceFit(*(psf.getSurfaceFit())) {};
      
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
      PSF(const MaskedImageT &trace,
          const std::vector<float> &xCenters,
          const std::vector<float> &spectrum,
          const std::vector<float> &spectrumVariance,
          const float xLow,
          const float xHigh,
          const unsigned int yLow,
          const unsigned int yHigh,
          const PTR(pfs::drp::stella::TwoDPSFControl) &twoDPSFControl,
          unsigned int iTrace = 0,
          unsigned int iBin = 0
      ) : _trace(trace),
          _twoDPSFControl(twoDPSFControl),
          _iTrace(iTrace),
          _iBin(iBin),
          _xLow(xLow),
          _xHigh(xHigh),
          _yLow(yLow),
          _yHigh(yHigh),
          _xCenters(xCenters),
          _spectrum(spectrum),
          _spectrumVariance(spectrumVariance),
          _imagePSF_XTrace(1),
          _imagePSF_YTrace(1),
          _imagePSF_ZTrace(1),
          _imagePSF_XRelativeToCenter(1),
          _imagePSF_YRelativeToCenter(1),
          _imagePSF_ZNormalized(1),
          _imagePSF_Weight(1),
          _pixelsFit(1),
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
      boost::shared_ptr<MaskedImageT> getTrace() const {return boost::shared_ptr<MaskedImageT>(new MaskedImageT(_trace));}
      unsigned int getIBin() const {return _iBin;}
      unsigned int getITrace() const {return _iTrace;}
      float getXLow() const {return _xLow;}
      float getXHigh() const {return _xHigh;}
      unsigned int getYLow() const {return _yLow;}
      unsigned int getYHigh() const {return _yHigh;}
      std::vector<float> getXCenters() const {return _xCenters;}
      std::vector<float> getSpectrum() const {return _spectrum;}
      std::vector<float> getSpectrumVariance() const {return _spectrumVariance;}
      std::vector<float> getImagePSF_XTrace() const {return _imagePSF_XTrace;}
      std::vector<float> getImagePSF_YTrace() const {return _imagePSF_YTrace;}
      std::vector<float> getImagePSF_ZTrace() const {return _imagePSF_ZTrace;}
      std::vector<float> getImagePSF_XRelativeToCenter() const {return _imagePSF_XRelativeToCenter;}
      std::vector<float> getImagePSF_YRelativeToCenter() const {return _imagePSF_YRelativeToCenter;}
      std::vector<float> getImagePSF_ZNormalized() const {return _imagePSF_ZNormalized;}
      std::vector<float> getImagePSF_Weight() const {return _imagePSF_Weight;}
      std::vector<float> getPixelsFit() const {return _pixelsFit;}
      bool isTwoDPSFControlSet() const {return _isTwoDPSFControlSet;}
      bool isPSFsExtracted() const {return _isPSFsExtracted;}
      PTR(SurfaceFit) getSurfaceFit() const {return boost::shared_ptr<SurfaceFit>(new SurfaceFit(_surfaceFit));}
      
      /// Return _2dPSFControl
      PTR(TwoDPSFControl) getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl);

      /// Return the SurfaceFit
//      PTR(SurfaceFit) getSurfaceFit() const {return boost::make_shared<SurfaceFit>(_surfaceFit);}

      bool extractPSFs();
      bool fitPSFKernel();
      bool calculatePSF();
  protected:

//    virtual std::string getPersistenceName() const;

//    virtual std::string getPythonModule() const;

//    virtual void write(OutputArchiveHandle & handle) const;

    private:
      MaskedImageT _trace;
      const PTR(TwoDPSFControl) _twoDPSFControl;
      const unsigned int _iTrace;
      const unsigned int _iBin;
      const float _xLow;
      const float _xHigh;
      const unsigned int _yLow;
      const unsigned int _yHigh;
      const std::vector<float> _xCenters;
      const std::vector<float> _spectrum;
      const std::vector<float> _spectrumVariance;
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
}}}
#endif
