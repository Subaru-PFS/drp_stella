#ifndef PFS_DRP_STELLA_PHOTOMETRY_H
#define PFS_DRP_STELLA_PHOTOMETRY_H

#include <ndarray_fwd.h>

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/SpectralPsf.h"

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
 * @param badBitMask : Bit mask for pixels to ignore.
 * @returns Catalog of flux measurements.
 */
lsst::afw::table::BaseCatalog photometer(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    pfs::drp::stella::SpectralPsf const& psf,
    lsst::afw::image::MaskPixel badBitMask=0x0
);


}}}  // namespace pfs::drp::stella



#endif