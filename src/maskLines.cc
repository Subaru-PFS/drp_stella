#include <numeric>

#include "ndarray.h"
#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/utils/math.h"


namespace pfs {
namespace drp {
namespace stella {


namespace {

/// Implementation of maskLines
///
/// Requires that the 'wavelength' and 'lines' arrays are both monotonic
/// increasing. This condition is not checked.
ndarray::Array<bool, 1, 1> maskLinesImpl(
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& lines,
    int maskRadius
) {
    std::ptrdiff_t const numWl = wavelength.size();
    ndarray::Array<bool, 1, 1> mask = ndarray::allocate(numWl);
    mask.deep() = false;
    std::size_t const numLines = lines.size();
    if (numLines == 0) {
        return mask;
    }

    std::ptrdiff_t wlIndex = 0;
    std::size_t linesIndex = 0;

    // Lines that are off-scale low in wavelength
    {
        double const wl0 = wavelength[0];
        for (; lines[linesIndex] < wl0 && linesIndex < numLines; ++linesIndex);  // increment only
        if (linesIndex > 0) {
            double const wlLine = lines[linesIndex - 1];  // wavelength of the closest line
            double const dispersion = wavelength[1] - wavelength[0];
            double const index = (wlLine - wavelength[0])/dispersion + maskRadius;
            std::ptrdiff_t const high = index + 0.5;
            if (high > 0) {
                mask[ndarray::view(0, high)] = true;
            }
        }
    }

    // linesIndex now refers to the first line greater than wavelength[0].

    for (; wlIndex < numWl && linesIndex < numLines; ++linesIndex) {
        double const wlLine = lines[linesIndex];
        // Bracket the line
        for (; wavelength[wlIndex] < wlLine && wlIndex < numWl; ++wlIndex);  // increment only
        if (wlIndex >= numWl) {
            // We're at the end; no need to keep checking for all lines
            break;
        }
        // Mask pixels within range
        std::ptrdiff_t const low = std::max(0L, wlIndex - maskRadius);
        std::ptrdiff_t const high = std::min(numWl, wlIndex + maskRadius);
        mask[ndarray::view(low, high)] = true;
    }

    // Lines that are off-scale high in wavelength
    if (linesIndex < numLines) {
        double const wlLine = lines[linesIndex];  // wavelength of the closest line
        double const dispersion = wavelength[numWl - 1] - wavelength[numWl - 2];
        double const index = numWl + (wlLine - wavelength[numWl - 1])/dispersion - maskRadius;
        std::ptrdiff_t const low = index - 0.5;
        if (low < numWl) {
            mask[ndarray::view(low, numWl)] = true;
        }
    }

    return mask;
}


}  // anonymous namespace


ndarray::Array<bool, 1, 1> maskLines(
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& lines,
    int maskRadius,
    bool sortedLines
) {
    // Verify/force wavelength+lines to both be monotonic increasing
    // wavelength: if monotonic decreasing, copy in reverse order
    // lines: sort
    std::size_t const numWavelength = wavelength.size();
    if (numWavelength < 2) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Wavelength array too short");
    }

    ndarray::Array<double, 1, 1> useWavelength;
    ndarray::Array<double, 1, 1> useLines;
    if (sortedLines) {
        useLines = lines;
    } else {
        useLines = ndarray::copy(lines);
        std::sort(useLines.begin(), useLines.end());
    }

    // Invert wavelength array, if necessary
    bool reversed = false;
    if (wavelength[0] > wavelength[numWavelength - 1]) {
        // Decreasing: reverse order.
        useWavelength = utils::reversed(wavelength);
        reversed = true;
    } else {
        useWavelength = wavelength;
    }

    // Check wavelength is monotonic increasing
    double last = -std::numeric_limits<double>::infinity();
    for (std::size_t ii = 0; ii < numWavelength; ++ii) {
        if (useWavelength[ii] < last) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              std::string("Wavelength array is not monotonic ") +
                              (reversed ? "decreasing" : "increasing"));
        }
        last = useWavelength[ii];
    }

    ndarray::Array<bool, 1, 1> mask = maskLinesImpl(useWavelength, useLines, maskRadius);

    ndarray::Array<bool, 1, 1> useMask;
    if (reversed) {
        useMask = utils::reversed(mask);
    } else {
        useMask = mask;
    }

    return useMask;
}


}}}  // namespace pfs::drp::stella
