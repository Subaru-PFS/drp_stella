#ifndef PFS_DRP_STELLA_PROFILE_H
#define PFS_DRP_STELLA_PROFILE_H

#include <vector>
#include <utility>

#include "ndarray_fwd.h"
#include "lsst/afw/image/MaskedImage.h"

namespace pfs {
namespace drp {
namespace stella {


/// Results from fitProfiles
struct FitProfilesResults {
    ndarray::Array<double, 3, 1> profiles; ///< Fiber profiles: 1d profile per fiber per swath
    ndarray::Array<bool, 3, 1> masks; ///< Mask for profiles
    std::vector<std::shared_ptr<lsst::afw::image::Image<float>>> backgrounds; ///< Background images

    FitProfilesResults(
        ndarray::Array<double, 3, 1> const& profiles_,
        ndarray::Array<bool, 3, 1> const& masks_,
        std::vector<std::shared_ptr<lsst::afw::image::Image<float>>> const& backgrounds_
    ) : profiles(profiles_), masks(masks_), backgrounds(backgrounds_) {}
};


/// Fit multiple fiber profiles for a swath
///
/// We perform a simultaneous fit of fiber profiles to multiple images.
///
/// @param images : Images to fit
/// @param centers : Fiber centers for each row, fiber, image.
/// @param spectra : Fiber spectra for each row, fiber, image.
/// @param fiberIds : Fiber IDs
/// @param ySwaths : y positions of swath centers
/// @param badBitMask : Mask bits to ignore
/// @param oversample : Oversampling factor
/// @param radius : Radius of fiber profile
/// @param sigma : Estimate of gaussian sigma of fiber profile
/// @param bgSize : Size of background super-pixels
/// @param rejIter : Number of rejection iterations
/// @param rejThresh : Rejection threshold (sigma)
/// @param matrixTol : Matrix solver tolerance
/// @return fiber profiles and mask
FitProfilesResults fitProfiles(
    std::vector<lsst::afw::image::MaskedImage<float>> const& images,
    std::vector<ndarray::Array<double, 2, 1>> const& centers,
    std::vector<ndarray::Array<float, 2, 1>> const& spectra,
    ndarray::Array<int, 1, 1> const& fiberIds,
    ndarray::Array<float, 1, 1> const& ySwaths,
    lsst::afw::image::MaskPixel badBitMask,
    int oversample,
    int radius,
    float sigma,
    lsst::geom::Extent2I const& bgSize,
    int rejIter=1,
    float rejThresh=4.0,
    float matrixTol=1e-4
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
