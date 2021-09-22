#ifndef PFS_DRP_STELLA_MAKEFOOTPRINT_H
#define PFS_DRP_STELLA_MAKEFOOTPRINT_H


#include "lsst/geom/Point.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/image/Image.h"

namespace pfs {
namespace drp {
namespace stella {


/// Make a Footprint for an emission line
///
/// This is a braindead deblender for use in centroiding lines. We need to
/// identify pixels that belong to the emission line of interest. We do this
/// by starting at the peak, and working up and down (in the spectral
/// dimension) to find the saddle point, which indicates the transition from
/// the peak of interest to another line. For left and right (the spatial
/// dimension), we simply apply the provided width (the distance between fibers
/// is well known, so there's no need for us to find it).
///
/// @param image : image to inspect
/// @param peak : integer pixel location of the peak
/// @param height : search height (should be PSF half-size + 1 in order for the
///     inner 3x3 pixels to be fully convolved by the PSF)
/// @param width : width of the Footprint (half the distance between fibers)
/// @return footprint for the emission line
lsst::afw::detection::Footprint makeFootprint(
    lsst::afw::image::Image<float> const& image,
    lsst::geom::Point2I const& peak,
    int height,
    float width
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
