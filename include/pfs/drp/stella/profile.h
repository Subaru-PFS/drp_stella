#ifndef PFS_DRP_STELLA_PROFILE_H
#define PFS_DRP_STELLA_PROFILE_H

#include <vector>
#include <utility>

#include "ndarray_fwd.h"
#include "lsst/afw/image/MaskedImage.h"

namespace pfs {
namespace drp {
namespace stella {


/// Fit multiple fiber profiles for a swath
///
/// We perform a simultaneous fit of fiber profiles to multiple images.
///
/// @param images : Images to fit
/// @param centers : Fiber centers for each row, fiber, image.
/// @param spectra : Fiber spectra for each row, fiber, image.
/// @param fiberIds : Fiber IDs
/// @param yMin : Minimum row to fit
/// @param yMax : Maximum row to fit
/// @param badBitMask : Mask bits to ignore
/// @param oversample : Oversampling factor
/// @param radius : Radius of fiber profile
/// @param rejIter : Number of rejection iterations
/// @param rejThresh : Rejection threshold (sigma)
/// @return fiber profiles and mask
std::pair<ndarray::Array<double, 2, 1>, ndarray::Array<bool, 2, 1>>
fitSwathProfiles(
    std::vector<lsst::afw::image::MaskedImage<float>> const& images,
    std::vector<ndarray::Array<double, 2, 1>> const& centers,
    std::vector<ndarray::Array<float, 2, 1>> const& spectra,
    ndarray::Array<int, 1, 1> const& fiberIds,
    int yMin,
    int yMax,
    lsst::afw::image::MaskPixel badBitMask,
    int oversample,
    int radius,
    int rejIter=1,
    float rejThresh=4.0
);


/// Fit Gaussian amplitudes as a function of row for multiple fibers
///
/// Simultaneously fits Gaussian profiles to multiple fibers in the image.
///
/// @param image : Image to fit
/// @param centers : Fiber centers for each row, fiber
/// @param sigma : Gaussian sigma of fiber profiles
/// @param badBitMask : Mask bits to ignore
/// @param nSigma : Maximum extent of fiber profile (multiple of sigma)
/// @return fiber amplitudes
ndarray::Array<double, 2, 1>
fitAmplitudes(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<double, 2, 1> const& centers,
    float sigma,
    lsst::afw::image::MaskPixel badBitMask,
    float nSigma=4.0
);


/// Calculate the profile for a swath
///
/// When measuring a fiber profile, we have a 2D numpy MaskedArray of values
/// (measurements of the profile for each row) which we want to collapse to a
/// 1D average profile.
///
/// @param values : Measurement values
/// @param mask : Measurement mask (true = bad value)
/// @param rejIter : Number of rejection iterations
/// @param, rejThresh : Rejection threshold (sigma)
/// @return collapsed values and mask
std::pair<ndarray::Array<double, 1, 1>, ndarray::Array<bool, 1, 1>>
calculateSwathProfile(
    ndarray::Array<double, 2, 1> const& values,
    ndarray::Array<bool, 2, 1> const& mask,
    int rejIter=1,
    float rejThresh=3.0
);


}}} // namespace pfs::drp::stella

#endif
