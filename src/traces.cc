#include "pfs/drp/stella/traces.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {

// State of finding peaks
//
// Used for findTracePeaks.
enum PeakState {
    NONE,  // We're not tracking a peak
    BEFORE,  // We're ascending to a peak
    AFTER,  // We're descending from a peak
};

} // anonymous namespace


std::vector<std::vector<std::shared_ptr<TracePeak>>> findTracePeaks(
    lsst::afw::image::MaskedImage<float> const& image,
    float threshold,
    lsst::afw::image::MaskPixel badBitMask
) {
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    float const threshold2 = std::pow(threshold, 2);
    std::vector<std::vector<std::shared_ptr<TracePeak>>> peaks{height};
    for (int yy = 0; yy < int(height); ++yy) {
        auto iter = image.row_begin(yy);
        auto & rowPeaks = peaks[yy];
        rowPeaks.reserve(width/4);
        PeakState state = NONE;
        int low = -1;
        int peak = -1;
        float lastSigNoise2 = 0;
        for (int xx = 0; xx < int(width); ++xx, ++iter) {
            if (iter.mask() & badBitMask) {
                // Masked pixel means we don't change state
                continue;
            }
            float const sigNoise2 = std::pow(iter.image(), 2)/iter.variance();
            if (sigNoise2 > threshold2) {
                switch (state) {
                  case NONE:
                    // We've found a trace
                    state = BEFORE;
                    low = xx;
                    break;
                  case BEFORE:
                    if (sigNoise2 < lastSigNoise2) {
                        // We found the peak
                        state = AFTER;
                        peak = xx - 1;
                    }
                    break;
                  case AFTER:
                    if (sigNoise2 > lastSigNoise2) {
                        // We found the end of this trace and the start of the next
                        rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, xx - 1));
                        state = BEFORE;
                        low = xx;
                        peak = -1;
                    }
                    break;
                }
            } else {
                switch (state) {
                  case BEFORE:
                    // The last pixel must be the peak
                    rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, xx - 1, xx - 1));
                    state = NONE;
                    low = -1;
                    peak = -1;
                    break;
                  case AFTER:
                    // We found the end of this trace
                    rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, xx - 1));
                    state = NONE;
                    low = -1;
                    peak = -1;
                    break;
                  case NONE:
                    // Nothing changed
                    break;
                }
            }
            lastSigNoise2 = sigNoise2;
        }
        // Forcibly end the search
        switch (state) {
          case BEFORE:
            // Last pixel must be the peak
            rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, width - 1, width - 1));
            break;
          case AFTER:
            // We found the end of this trace
            rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, width - 1));
            break;
          case NONE:
            // Nothing changed
            break;
        }
    }
    return peaks;
}


void centroidTrace(
    std::vector<std::shared_ptr<TracePeak>> & peaks,
    lsst::afw::image::MaskedImage<float> const& image,
    int radius,
    lsst::afw::image::MaskPixel badBitMask
) {
    for (auto & pp : peaks) {
        // Going with a basic centroid for now, since RHL's book says that's the best for high S/N.
        // Later, might try a Gaussian fit, or SdssCentroid's Gaussian quartic correction.
        int const xMin = std::max(int(pp->peak - radius), 0);
        int const xMax = std::min(int(pp->peak + radius + 0.5), image.getWidth() - 1);
        double xSum = 0.0;
        double sum = 0.0;
        auto iter = image.row_begin(pp->span.getY()) + xMin;
        for (int xx = xMin; xx <= xMax; ++xx, ++iter) {
            if (iter.mask() & badBitMask) continue;
            sum += iter.image();
            xSum += xx*iter.image();
        }
        pp->peak = xSum/sum;
    }
}


}}} // namespace pfs::drp::stella
