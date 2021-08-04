#ifndef PFS_DRP_STELLA_MASKLINES_H
#define PFS_DRP_STELLA_MASKLINES_H

#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {


/// Mask lines in a spectrum
///
/// @param wavelength : wavelength array of spectrum. Must be monotonic. If not monotonic increasing, will be
///     temporarily sorted, so it's more efficient if this is monotonic increasing.
/// @param lines : wavelengths of lines to mask
/// @param maskRadius : number of pixels either side of the line to mask
/// @param sortedLines : are the line wavelengths sorted? If not, it will be temporarily sorted, so setting
///     sortedLines=true is more efficient. If sortedLines=true but the list is not actually sorted, the
///     result will be incorrect.
/// @return boolean array indicating whether a line is within the masking radius.
ndarray::Array<bool, 1, 1> maskLines(
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& lines,
    int maskRadius,
    bool sortedLines=false
);


}}}  // namespace pfs::drp::stella


#endif  // include guard
