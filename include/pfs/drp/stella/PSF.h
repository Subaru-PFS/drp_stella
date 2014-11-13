#if !defined(PFS_DRP_STELLA_FIBERTRACES_H)
#define PFS_DRP_STELLA_FIBERTRACES_H

#include <vector>
#include <iostream>
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
#include "SurfaceFit.h"
//#include "lsst/afw/detection/Psf.h"

#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"

#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella {
  
  template<typename ImageT, typename MaskT=afwImage::MaskPixel, typename VarianceT=afwImage::VariancePixel>
  class PSF {
    public:
      typedef afwImage::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
      
      /**
       *  @brief Constructor for a PSF
       *
       *  @param[in] width   Number of columns in realizations of the PSF at a point.
       *  @param[in] height  Number of rows in realizations of the PSF at a point.
       */
      PSF(int width,
          int height);
      
      PSF(const MaskedImageT &trace,
          const int yLow_In,
          const int yHigh_In);
      
      /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
      virtual PTR(PSF) clone() const;
      
      /// Return the dimensions of the images returned by computeImage()
      geom::Extent2I getDimensions() const { return _dimensions; }
      
      /// Whether the Psf is persistable; always true.
      virtual bool isPersistable() const { return true; }
      
      /// Return _2dPSFControl
      PTR(TwoDPSFControl) getTwoDPSFControl() const { return _twoDPSFControl; }
  
      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(PTR(TwoDPSFControl) twoDPSFControl);
      
    protected:
    
//    virtual std::string getPersistenceName() const;
    
//    virtual std::string getPythonModule() const;
    
//    virtual void write(OutputArchiveHandle & handle) const;
    
    private: 
      afwImage::Image<ImageT> _psf;
      SurfaceFit
      PTR(TwoDPSFControl) _twoDPSFControl;
      geom::Extent2I _dimensions;
      int _iBin;
      std::vector<float> _imagePSF_XTrace;
      std::vector<float> _imagePSF_YTrace;
      std::vector<float> _imagePSF_ZTrace;
      std::vector<float> _imagePSF_XRelativeToCenter;
      std::vector<float> _imagePSF_YRelativeToCenter;
      std::vector<float> _imagePSF_ZNormalized;
      std::vector<float> _imagePSF_Weight;
      bool _isTwoDPSFControlSet;
      
      bool computeKernelImage();
  }
}}}
#endif
