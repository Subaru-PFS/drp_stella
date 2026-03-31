#ifndef PFS_DRP_STELLA_WARP_H
#define PFS_DRP_STELLA_WARP_H

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/warpExposure.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


//@{
/// Warp a single fiber's trace to a rectified image
///
/// @param image : input image to warp
/// @param detectorMap : mapping between fiberId,wavelength and x,y on the detector
/// @param fiberId : fiber to warp
/// @param halfWidth : half-width of the output image in pixels
/// @param warpingKernelName : name of warping kernel to use (e.g. "lanczos3")
/// @return warped fiber image
lsst::afw::image::MaskedImage<float> warpFiber(
    lsst::afw::image::MaskedImage<float> const& image,
    pfs::drp::stella::DetectorMap const& detectorMap,
    int fiberId,
    int halfWidth,
    std::string const &warpingKernelName="lanczos3"
);
lsst::afw::image::MaskedImage<float> warpFiber(
    lsst::afw::image::MaskedImage<float> const& image,
    pfs::drp::stella::DetectorMap const& detectorMap,
    int fiberId,
    int halfWidth,
    lsst::afw::math::WarpingControl const& ctrl,
    lsst::afw::image::MaskedImage<float>::SinglePixel const& pad
);
//@}


//@{
/// Warp an image to match a different detector map
/// @param fromImage : input image to warp
/// @param fromDetectorMap : mapping between fiberId,wavelength and x,y on the input image's detector
/// @param toDetectorMap : mapping between fiberId,wavelength and x,y on the output image's detector
/// @param badBitmask : bitmask to apply to bad pixels in the output image
/// @param warpingKernelName : name of warping kernel to use (e.g. "lanczos3")
/// @param numWavelengthKnots : number of knots to use in the wavelength dimension when interpolating
/// @return warped image
lsst::afw::image::MaskedImage<float> warpImage(
    lsst::afw::image::MaskedImage<float> const& fromImage,
    pfs::drp::stella::DetectorMap const& fromDetectorMap,
    pfs::drp::stella::DetectorMap const& toDetectorMap,
    std::string const &warpingKernelName="lanczos3",
    int numWavelengthKnots=75
);
lsst::afw::image::MaskedImage<float> warpImage(
    lsst::afw::image::MaskedImage<float> const& fromImage,
    pfs::drp::stella::DetectorMap const& fromDetectorMap,
    pfs::drp::stella::DetectorMap const& toDetectorMap,
    lsst::afw::math::WarpingControl const& ctrl,
    lsst::afw::image::MaskedImage<float>::SinglePixel const& pad,
    int numWavelengthKnots=75
);


}}}  // namespace pfs::drp::stella


#endif  // include guard
