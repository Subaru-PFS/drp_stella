#ifndef PFS_DRP_STELLA_CENTROIDIMAGE_H
#define PFS_DRP_STELLA_CENTROIDIMAGE_H

#include "lsst/geom/Point.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image.h"

namespace pfs {
namespace drp {
namespace stella {


/// Find peak on image
///
/// This function finds the peak of an image, starting from a given point.
///
/// @param image : Image to search.
/// @param center : Starting point for search.
/// @param halfWidth : Half-width of search box.
/// @param badBitMask : Mask plane bitmask to ignore.
/// @returns Peak of image.
lsst::geom::Point2I findPeak(
    lsst::afw::image::MaskedImage<float> const& image,
    lsst::geom::Point2D const& center,
    int halfWidth,
    lsst::afw::image::MaskPixel badBitMask=0
);


/// Centroid an exposure
///
/// This provides a simple interface to the SdssCentroid measurement algorithm
/// from LSST to measure a single centroid.
///
/// This function is the one that does the main work of centroiding.
///
/// Not templated, because the measurement framework only works on float
/// exposures.
///
/// @param exposure : Exposure containing image to measure and (approximate) PSF
///     model.
/// @param point : Estimate of centroid, to be refined.
/// @returns Centroid calculated with SdssCentroid measurement algorithm.
lsst::geom::Point2D centroidExposure(
    lsst::afw::image::Exposure<float> const& exposure,
    lsst::geom::Point2D const& point
);


/// Centroid an image
///
/// This provides a simple interface to the SdssCentroid measurement algorithm
/// from LSST to measure a single centroid.
///
/// This overload creates an Exposure<float> with a blank mask plane, measures
/// an approximate centroid using first moments (over the entire image, so it
/// should contain only a single object to be centroided) and passes it to the
/// overload that does the actual centroiding.
///
/// @param image : Image to measure.
/// @param psf : Approximate point-spread function.
/// @returns Centroid calculated with SdssCentroid measurement algorithm.
template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    std::shared_ptr<lsst::afw::detection::Psf> psf
);


/// Centroid an image
///
/// This provides a simple interface to the SdssCentroid measurement algorithm
/// from LSST to measure a single centroid.
///
/// This overload creates an Exposure<float> with a blank mask plane, measures
/// an approximate centroid using first moments (over the entire image, so it
/// should contain only a single object to be centroided and little noise),
/// generates a Gaussian and passes it to the overload that does the actual
/// centroiding.
///
/// @param image : Image to measure.
/// @param psfSigma : Approximate Gaussian sigma of the point-spread function.
/// @param nSigma : Number of sigma (either side) for PSF kernel size.
/// @returns Centroid calculated with SdssCentroid measurement algorithm.
template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    float psfSigma,
    float nSigma=3
);


/// Centroid an image
///
/// This provides a simple interface to the SdssCentroid measurement algorithm
/// from LSST to measure a single centroid.
///
/// This overload creates an Exposure<float> with a blank mask plane, measures
/// an approximate centroid using first moments and Gaussian PSF using second
/// moments (over the entire image, so it should contain only a single object to
/// be centroided and no noise) and passes it to the overload that does the
/// actual centroiding.
///
/// @param image : Image to measure.
/// @param nSigma : Number of sigma (either side) for PSF kernel size.
/// @returns Centroid calculated with SdssCentroid measurement algorithm.
template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    float nSigma=3
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
