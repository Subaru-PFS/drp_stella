#ifndef PFS_DRP_STELLA_TRACES_H
#define PFS_DRP_STELLA_TRACES_H

#include <map>
#include <vector>
#include <limits>
#include "ndarray_fwd.h"
#include "lsst/afw/geom/SpanSet.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


/// A peak found in the trace
struct TracePeak {
    lsst::afw::geom::Span const span;  ///< Span containing the peak
    double peak;  ///< Peak position in the spatial dimension
    double peakErr;  ///< Error in peak position
    float flux;  ///< Flux measurement
    float fluxErr;  ///< Error in flux measurement

    /// Constructor
    ///
    /// @param row : Row containing the peak
    /// @param low : Column at the low-end of the span containing the peak
    /// @param peak_ : Peak position in the spatial dimension
    /// @param high : Column at the high-end of the span containing the peak
    /// @param peakErr_ : Error in peak position
    /// @param flux_ : Flux measurement
    /// @param fluxErr_ : Error in flux measurement
    TracePeak(int row, int low, double peak_, int high,
              double peakErr_=std::numeric_limits<double>::quiet_NaN(),
              float flux_=std::numeric_limits<float>::quiet_NaN(),
              float fluxErr_=std::numeric_limits<float>::quiet_NaN()) :
      span(row, low, high),
      peak(peak_),
      peakErr(peakErr_),
      flux(flux_),
      fluxErr(fluxErr_)
      {}

    TracePeak(TracePeak const&) = delete;
    TracePeak(TracePeak &&) = default;
    TracePeak & operator=(TracePeak const&) = delete;
    TracePeak & operator=(TracePeak &&) = delete;
};


std::ostream& operator<<(std::ostream& os, TracePeak const& tp);


/// Find peaks on an image
///
/// We find all peaks above threshold in each row, for later association.
///
/// @param image : Image to search for peaks
/// @param threshold : Threshold value for peaks
/// @param badBitMask : Bitmask to apply to identify bad pixels. Bad pixels count as a pixel below threshold.
/// @return a list of peaks for each row of the image
std::vector<std::vector<std::shared_ptr<TracePeak>>> findTracePeaks(
    lsst::afw::image::MaskedImage<float> const& image,
    float threshold,
    lsst::afw::image::MaskPixel badBitMask=0
);


/// Find peaks on an image around a detectorMap
///
/// @param image : Image to search for peaks
/// @param detectorMap : Mapping of fiberId,wavelength --> x,y
/// @param threshold : Threshold value for peaks
/// @param radius : Radius around expected peak to search
/// @param badBitMask : Bitmask to apply to identify bad pixels. Bad pixels count as a pixel below threshold.
/// @param fiberId : Fiber identifiers of interest
/// @return a list of peaks indexed by fiber identifier
std::map<int, std::vector<std::shared_ptr<TracePeak>>> findTracePeaks(
    lsst::afw::image::MaskedImage<float> const& image,
    DetectorMap const& detectorMap,
    float threshold,
    float radius,
    lsst::afw::image::MaskPixel badBitMask=0,
    ndarray::Array<int, 1, 1> const& fiberId=ndarray::Array<int, 1, 1>()
);


/// Centroid peak
///
/// We measure the centroid for all peaks, modifying the input
///
/// @param peak : Peak to centroid; modified
/// @param image : Image on which to measure centroids
/// @param psfSigma : Gaussian sigma of the PSF in the x dimension
/// @param badBitMask : Bitmask to apply to identify bad pixels. Bad pixels are ignored in the measurement
/// @param extent : Size of convolution kernel, relative to psfSigma
/// @param ampAst4 : Amplitude of fourth-order correction compared to theory; 4/3 used for SDSS
void centroidPeak(
    TracePeak & peak,  // modified
    lsst::afw::image::MaskedImage<float> const& image,
    float psfSigma,
    lsst::afw::image::MaskPixel badBitMask=0,
    float extent=3.0,
    float ampAst4=1.33
);


}}} // namespace pfs::drp::stella

#endif
