#if !defined(PFS_DRP_STELLA_PSF_H)
#define PFS_DRP_STELLA_PSF_H

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
#include "FiberTraces.h"
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
       *  @param[in] trace              Masked image of a FiberTrace for which to compute PSF in one bin specified by yLow_In and yHigh_In
       *  @param[in] xCenters           Vector containing the xCenters of the FiberTrace
       *  @param[in] yLow               Lower y limit of Bin for which the PSF shall be computed
       *  @param[in] yHigh              Upper y limit of Bin for which the PSF shall be computed
       *  @param[in] twoDPSFControl     Structure containing the parameters for the computation of the PSF
       *  @param[in] iBin               Bin number for which the PSF shall be computed (for debugging purposes only)
       */
      PSF(const MaskedImageT &trace,
          const std::vector<float> &xCenters,
          const float xLow,
          const float xHigh,
          const unsigned int yLow,
          const unsigned int yHigh,
          const pfs::drp::stella::TwoDPSFControl &twoDPSFControl,
          int iBin = 0
         );

      /// Polymorphic deep copy; should usually be unnecessary because Psfs are immutable.
//      virtual PTR(PSF) clone() const;

      /// Return the dimensions of the images returned by computeImage()
//      geom::Extent2I getDimensions() const { return _dimensions; }

      /// Whether the Psf is persistable; always true.
//      virtual bool isPersistable() const { return true; }

      /// Return _2dPSFControl
      TwoDPSFControl getTwoDPSFControl() const { return _twoDPSFControl; }

      /// Set the _twoDPSFControl
      bool setTwoDPSFControl(TwoDPSFControl &twoDPSFControl);

      /// Return the SurfaceFit
      PTR(SurfaceFit) getSurfaceFit() const {return boost::make_shared<SurfaceFit>(_surfaceFit);}

      void calculatePSFKernel(blitz::Array<double, 2> &PSF2D_Out);
  protected:

//    virtual std::string getPersistenceName() const;

//    virtual std::string getPythonModule() const;

//    virtual void write(OutputArchiveHandle & handle) const;

    private:
//      afwImage::Image<ImageT> _psf;
      const TwoDPSFControl _twoDPSFControl;
      SurfaceFit _surfaceFit;
      const unsigned int _iBin;
      const float _xLow;
      const float _xHigh;
      const unsigned int _yLow;
      const unsigned int _yHigh;
      const std::vector<float> _xCenters;
      std::vector<float> _imagePSF_XTrace;
      std::vector<float> _imagePSF_YTrace;
      std::vector<float> _imagePSF_ZTrace;
      std::vector<float> _imagePSF_XRelativeToCenter;
      std::vector<float> _imagePSF_YRelativeToCenter;
      std::vector<float> _imagePSF_ZNormalized;
      std::vector<float> _imagePSF_Weight;
      bool _isTwoDPSFControlSet;

  };
}}}
#endif
