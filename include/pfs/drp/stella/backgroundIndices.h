#ifndef PFS_DRP_STELLA_BACKGROUND_H
#define PFS_DRP_STELLA_BACKGROUND_H

#include "ndarray_fwd.h"
#include "lsst/geom/Extent.h"
#include "lsst/afw/image/Image.h"

namespace pfs {
namespace drp {
namespace stella {


/// Return the number of super-pixels
///
/// @param dims : Image dimensions
/// @param bgSize : Size of super-pixels (pixels)
/// @return number of super-pixels
int getNumBackgroundIndices(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize
);


/// Return an image mapping pixel to super-pixel index
///
/// Super-pixels are centered on the image.
///
/// @param dims : Image dimensions
/// @param bgSize : Size of super-pixels (pixels)
/// @param indexOffset : Offset to add to index
/// @return image mapping pixel to super-pixel index
ndarray::Array<int, 2, 1> calculateBackgroundIndices(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize,
    int indexOffset=0
);


/// Construct an image of the background super-pixels
///
/// This is an image solely of the under-sampled super-pixels. To scale it up
/// to match the original image dimensions, use lsst::afw::math::BackgroundMI.
///
/// @param dims : Image dimensions
/// @param bgSize : Size of super-pixels (pixels)
/// @param values : Values for each super-pixel
/// @param indexOffset : Offset to add to index
/// @return image of the background super-pixels
template<typename ValueT>
std::shared_ptr<lsst::afw::image::Image<float>> makeBackgroundImage(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize,
    ndarray::Array<ValueT, 1, 1> const& values,
    int indexOffset
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
