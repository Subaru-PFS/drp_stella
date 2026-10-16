#ifndef PFS_DRP_STELLA_PHOTOMETRY_H
#define PFS_DRP_STELLA_PHOTOMETRY_H

#include <ndarray_fwd.h>

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Psf.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


/** Photometer the nominated lines in an image
 *
 * We perform a simultaneous fit of PSFs to the image.
 *
 * The returned catalog columns are:
 * - fiberId (int) : Fiber identifier of trace containing the line.
 * - wavelength (double) : Wavelength of line.
 * - flag (bool) : Is the measurement bad?
 * - flux (double) : Measured flux of line.
 * - fluxErr (double) : Statistical variance in measured flux of line. This is
 *   calculated using the image variances, and does not include any covariance
 *   with neighbouring lines.
 *
 * @param image : Image containing lines to measure.
 * @param fiberId : Fiber identifiers of traces for which to measure lines.
 * @param wavelength : Wavelength (nm) of lines to measure.
 * @param psf : Point-spread function.
 * @param detectorMap : Mapping between fiberId,wavelength and x,y, used to
 *     compute positions when not supplied directly via ``positions``.
 * @param badBitMask : Bit mask for pixels to ignore.
 * @param positions : Positions at which to measure; if empty, use fiberId and
 *     wavelength and ``detectorMap``.
 * @returns Catalog of flux measurements.
 */
lsst::afw::table::BaseCatalog photometer(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    lsst::afw::detection::Psf const& psf,
    std::shared_ptr<DetectorMap> const& detectorMap,
    lsst::afw::image::MaskPixel badBitMask=0x0,
    ndarray::Array<double, 2, 1> const& positions=ndarray::Array<double, 2, 1>()
);


}}}  // namespace pfs::drp::stella



#endif